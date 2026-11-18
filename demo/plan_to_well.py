"""Toy bench task: move a pipette tip held by a planar 2-link arm from a tube rack
to a well of a 96-well plate, avoiding a rack wall and a reagent bottle.
Joint-space RRT plus shortcut smoothing. Pure numpy, CPU only. Top-down 2D view.
"""
import time
import numpy as np

L1, L2 = 0.30, 0.25                      # link lengths (m)
PLATE_ORIGIN = np.array([0.20, 0.05])     # center of well A1 (m)
PITCH = 0.009                             # SBS 96-well pitch: 9 mm
ROWS = "ABCDEFGH"
OBSTACLES = [                             # axis-aligned boxes: xmin, ymin, xmax, ymax
    (0.33, -0.08, 0.36, 0.12),            # tube rack wall
    (0.15, 0.27, 0.24, 0.34),             # reagent bottle
]
TUBE = np.array([-0.05, 0.40])            # tube position to aspirate from


def well_name(index, order="row"):
    """1-based well index -> name. 'row': A1..A12,B1..; 'col': A1..H1,A2.."""
    i = index - 1
    r, c = (i // 12, i % 12) if order == "row" else (i % 8, i // 8)
    return f"{ROWS[r]}{c + 1}"


def well_xy(name):
    r, c = ROWS.index(name[0]), int(name[1:]) - 1
    return PLATE_ORIGIN + np.array([c * PITCH, -r * PITCH])


def fk(q):
    e = np.array([L1 * np.cos(q[0]), L1 * np.sin(q[0])])
    t = e + np.array([L2 * np.cos(q[0] + q[1]), L2 * np.sin(q[0] + q[1])])
    return e, t


def ik(p, elbow=1):
    d = (p @ p - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(d) > 1:
        return None
    q2 = elbow * np.arccos(d)
    q1 = np.arctan2(p[1], p[0]) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return np.array([q1, q2])


def in_collision(q):
    e, t = fk(q)
    pts = np.vstack([np.linspace([0, 0], e, 15), np.linspace(e, t, 15)])
    for x0, y0, x1, y1 in OBSTACLES:
        if np.any((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)):
            return True
    return False


def edge_free(a, b, step=0.02):
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)))
    return not any(in_collision(a + (b - a) * s) for s in np.linspace(0, 1, n))


def rrt(start, goal, rng, iters=3000, step=0.15, goal_bias=0.1):
    nodes, parent = [start], [-1]
    arr = np.zeros((iters + 2, 2))
    arr[0] = start
    for _ in range(iters):
        x = goal if rng.random() < goal_bias else rng.uniform(-np.pi, np.pi, 2)
        i = int(np.argmin(np.linalg.norm(arr[: len(nodes)] - x, axis=1)))
        d = x - nodes[i]
        new = nodes[i] + d * min(1, step / (np.linalg.norm(d) + 1e-9))
        if edge_free(nodes[i], new):
            arr[len(nodes)] = new
            nodes.append(new)
            parent.append(i)
            if np.linalg.norm(new - goal) < step and edge_free(new, goal):
                nodes.append(goal)
                parent.append(len(nodes) - 2)
                path, k = [], len(nodes) - 1
                while k != -1:
                    path.append(nodes[k])
                    k = parent[k]
                return path[::-1]
    return None


def shortcut(path, rng, tries=100):
    path = list(path)
    for _ in range(tries):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i > 1 and edge_free(path[i], path[j]):
            path = path[: i + 1] + path[j:]
    return path


def length(path):
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])))


def main():
    print("well 65 ->", well_name(65, "row"), "(row-major),", well_name(65, "col"), "(column-major)")
    start = ik(TUBE)
    if start is None or in_collision(start):
        start = ik(TUBE, elbow=-1)
    assert start is not None and not in_collision(start)
    blocked = 0
    ok, times, raw, smooth = 0, [], [], []
    targets = ["A1", "A12", "D6", "E5", "F5", "H1", "H12", "C9", "G3", "B7"]
    trials = 0
    for w in targets:
        goal = ik(well_xy(w))
        if goal is None or in_collision(goal):
            goal = ik(well_xy(w), elbow=-1)
        blocked += not edge_free(start, goal)
        for seed in range(10):
            trials += 1
            rng = np.random.default_rng(seed)
            t0 = time.perf_counter()
            path = rrt(start, goal, rng)
            if path is None:
                continue
            sp = shortcut(path, rng)
            times.append(time.perf_counter() - t0)
            ok += 1
            raw.append(length(path))
            smooth.append(length(sp))
    if not ok:
        print("no plan found")
        return
    print(f"straight joint-space move blocked by obstacles for {blocked} of {len(targets)} wells")
    print(f"trials: {trials}, success: {ok} ({100 * ok / trials:.1f}%)")
    print(f"plan time: mean {np.mean(times) * 1000:.1f} ms, max {np.max(times) * 1000:.1f} ms")
    print(f"joint path length: raw {np.mean(raw):.2f} rad -> shortcut {np.mean(smooth):.2f} rad")


if __name__ == "__main__":
    main()

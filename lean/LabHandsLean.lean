/-!
# LabHandsLean: L1 Continuous Obstacle Clearance, Lipschitz Swept-Segment & Slosh Proofs

Formalizes the `lab-hands` (Layer L1) benchtop manipulation and trajectory time-scaling guardrails:
1. **Continuous Obstacle Clearance**: Swept arm segments maintain `>= 5 mm` (`5000 um`) safety margin.
2. **Kinematic Lipschitz Swept-Segment Soundness (`#29`)**: Proves that bounding finite joint-segment
   grid samples by `reqMarginUM + halfStepSweepUM` (with Jacobian displacement upper bound
   `tipSweepBoundUM = (L1 + L2) * |dq1| + L2 * |dq2|`) guarantees continuous clearance `>= reqMarginUM`
   at every interior point along `[0, 1]`.
3. **Self-Collision & Rim Transit Bounds**: `|q2| <= 2600 mrad` and `0 <= vesselRimClearZMM`.
4. **Open-Meniscus Pendulum-Slosh Minimum-Jerk Gate (Idea #19)**: Time-scaled trajectories
   keep predicted dynamic meniscus rise strictly below `0.7 * h_free`
   (`10 * predMeniscusRiseUM < 7 * freeHeadspaceUM`), guaranteeing zero well spills while
   maintaining `>= 5 mm` continuous obstacle clearance.
-/
namespace LabHandsLean

/-- Structured boundary violation reporting repository, rule identifier, and detail. -/
structure RegimeViolation where
  repo   : String
  rule   : String
  detail : String
  deriving DecidableEq, Repr

/-- Return the first violation in a list of `(holds, violation)` checks. -/
def firstViolation : List (Bool × RegimeViolation) → Option RegimeViolation
  | [] => none
  | (true, _) :: rest => firstViolation rest
  | (false, v) :: _ => some v

@[simp] theorem firstViolation_nil : firstViolation [] = none := rfl

@[simp] theorem firstViolation_cons (b : Bool) (v : RegimeViolation)
    (rest : List (Bool × RegimeViolation)) :
    firstViolation ((b, v) :: rest) = none ↔ b = true ∧ firstViolation rest = none := by
  cases b <;> simp [firstViolation]

theorem firstViolation_none_iff (checks : List (Bool × RegimeViolation)) :
    firstViolation checks = none ↔ ∀ p ∈ checks, p.1 = true := by
  induction checks with
  | nil => simp [firstViolation]
  | cons hd tl ih =>
    rcases hd with ⟨b, _⟩
    cases b with
    | false => simp [firstViolation]
    | true => simp [firstViolation, ih]

/-! ## Kinematic Lipschitz Swept-Segment Subdivision Soundness (`#29`) -/

/-- Exact Link-1 Jacobian displacement bound `L1 * |dq1|` in micrometers (`mm * mrad = um`). -/
def link1SweepBoundUM (l1MM dq1Mrad : Nat) : Nat :=
  l1MM * dq1Mrad

/-- Kinematic Jacobian displacement upper bound `(L1 + L2) * |dq1| + L2 * |dq2|` in micrometers
    (`1 mm * 1 mrad = 10^-3 m * 10^-3 rad = 10^-6 m = 1 um`), bounding every point on Links 1 and 2. -/
def tipSweepBoundUM (l1MM l2MM dq1Mrad dq2Mrad : Nat) : Nat :=
  (l1MM + l2MM) * dq1Mrad + l2MM * dq2Mrad

/-- Link-1 displacement is bounded above by the full 2-link `tipSweepBoundUM`. -/
theorem link1Sweep_le_tipSweepBoundUM (l1MM l2MM dq1Mrad dq2Mrad : Nat) :
    link1SweepBoundUM l1MM dq1Mrad ≤ tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad := by
  unfold link1SweepBoundUM tipSweepBoundUM
  have h1 : (l1MM + l2MM) * dq1Mrad = l1MM * dq1Mrad + l2MM * dq1Mrad := Nat.add_mul l1MM l2MM dq1Mrad
  omega

/-- Conservative ceiling bound on `sweepUM / (2 * nIntervals)` in micrometers for `nIntervals` sub-segments. -/
def halfStepSweepUM (sweepUM nIntervals : Nat) : Nat :=
  if nIntervals == 0 then sweepUM
  else (sweepUM + 2 * nIntervals - 1) / (2 * nIntervals)

/-- Ceiling division soundness: `2 * nIntervals * halfStepSweepUM` conservatively upper-bounds `sweepUM`. -/
theorem halfStepSweepUM_spec (sweepUM nIntervals : Nat) (hn : 0 < nIntervals) :
    sweepUM ≤ 2 * nIntervals * halfStepSweepUM sweepUM nIntervals := by
  have hne : (nIntervals == 0) = false := by
    cases nIntervals with
    | zero => omega
    | succ _ => rfl
  unfold halfStepSweepUM
  rw [hne]
  simp only [Bool.false_eq_true, ↓reduceIte]
  have hd : 0 < 2 * nIntervals := by omega
  have hdiv :
      (2 * nIntervals) * ((sweepUM + 2 * nIntervals - 1) / (2 * nIntervals))
        + (sweepUM + 2 * nIntervals - 1) % (2 * nIntervals)
      = sweepUM + 2 * nIntervals - 1 :=
    Nat.div_add_mod (sweepUM + 2 * nIntervals - 1) (2 * nIntervals)
  have hmod : (sweepUM + 2 * nIntervals - 1) % (2 * nIntervals) < 2 * nIntervals :=
    Nat.mod_lt (sweepUM + 2 * nIntervals - 1) hd
  omega

/-- Symmetric absolute difference `|a - b|` on natural grid coordinates. -/
def natAbsDiff (a b : Nat) : Nat :=
  if a ≤ b then b - a else a - b

theorem natAbsDiff_comm (a b : Nat) : natAbsDiff a b = natAbsDiff b a := by
  unfold natAbsDiff
  split <;> split <;> omega

/-- Constructive nearest sample index `k in 0..nIntervals` on a uniform grid of step `2 * res`. -/
def nearestGridIndex (res pos : Nat) : Nat :=
  (pos + res) / (2 * res)

/-- Grid-covering lemma: every scaled position `pos <= 2 * nIntervals * res` lies within `res`
    (half a grid sub-interval) of sample `nearestGridIndex res pos <= nIntervals`. -/
theorem nearestGridIndex_bounds (nIntervals res pos : Nat)
    (hres : 0 < res) (hpos : pos ≤ 2 * nIntervals * res) :
    nearestGridIndex res pos ≤ nIntervals ∧
      natAbsDiff (2 * nearestGridIndex res pos * res) pos ≤ res := by
  unfold nearestGridIndex
  have htwoR : 0 < 2 * res := by omega
  have hdiv : (2 * res) * ((pos + res) / (2 * res)) + (pos + res) % (2 * res) = pos + res :=
    Nat.div_add_mod (pos + res) (2 * res)
  have hrem : (pos + res) % (2 * res) < 2 * res :=
    Nat.mod_lt (pos + res) htwoR
  have hmul_eq : (2 * res) * ((pos + res) / (2 * res)) = 2 * ((pos + res) / (2 * res)) * res := by
    rw [Nat.mul_assoc, Nat.mul_comm res ((pos + res) / (2 * res)), ← Nat.mul_assoc]
  have hk : (pos + res) / (2 * res) ≤ nIntervals := by
    if hle : (pos + res) / (2 * res) ≤ nIntervals then
      exact hle
    else
      have hk1 : nIntervals + 1 ≤ (pos + res) / (2 * res) := by omega
      have hmul_le : (2 * res) * (nIntervals + 1) ≤ (2 * res) * ((pos + res) / (2 * res)) :=
        Nat.mul_le_mul_left (2 * res) hk1
      have hexpand : (2 * res) * (nIntervals + 1) = 2 * nIntervals * res + 2 * res := by
        rw [Nat.mul_add, Nat.mul_one, Nat.mul_assoc, Nat.mul_comm res nIntervals, ← Nat.mul_assoc]
      omega
  refine ⟨hk, ?_⟩
  unfold natAbsDiff
  split <;> omega

/-- Positive integer factor cancellation: `0 < d` and `d * x <= d * y` imply `x <= y`. -/
theorem int_le_of_pos_mul_le (d x y : Int) (hd : 0 < d) (h : d * x ≤ d * y) : x ≤ y := by
  if hxy : x ≤ y then
    exact hxy
  else
    have hdiff_pos : 0 < x - y := by omega
    have hpos : 0 < d * (x - y) := Int.mul_pos hd hdiff_pos
    rw [Int.mul_sub] at hpos
    omega

/-- **Continuous Swept-Segment Lipschitz Soundness Theorem (`#29`)**:
    Along a joint-space chord `(q_a, q_b)` subdivided into `nIntervals > 0` equal steps,
    let `sweepUM = tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad` be the kinematic Lipschitz bound.
    If `clearanceUM` obeys the Lipschitz bound on `[0, 2 * nIntervals * res]` for any resolution `res > 0`,
    and every grid sample `k in 0..nIntervals` satisfies
    `reqMarginUM + halfStepSweepUM sweepUM nIntervals <= clearanceUM (2 * k * res)`,
    then EVERY continuous point `pos <= 2 * nIntervals * res` satisfies `reqMarginUM <= clearanceUM pos`. -/
theorem sweptSegment_continuous_clearance_sound
    (l1MM l2MM dq1Mrad dq2Mrad nIntervals res reqMarginUM : Nat)
    (clearanceUM : Nat → Int)
    (hn : 0 < nIntervals)
    (hres : 0 < res)
    (hLip : ∀ p q : Nat,
      ((2 * nIntervals * res : Nat) : Int) * (clearanceUM p - clearanceUM q) ≤
        (tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad : Int) * (natAbsDiff p q : Int))
    (hGrid : ∀ k : Nat, k ≤ nIntervals →
      ((reqMarginUM + halfStepSweepUM (tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad) nIntervals : Nat) : Int) ≤
        clearanceUM (2 * k * res)) :
    ∀ pos : Nat, pos ≤ 2 * nIntervals * res → (reqMarginUM : Int) ≤ clearanceUM pos := by
  intro pos hpos
  have hD_nat : 0 < 2 * nIntervals * res := Nat.mul_pos (by omega) hres
  have hD_pos : (0 : Int) < ((2 * nIntervals * res : Nat) : Int) := by omega
  have ⟨hk, hdist⟩ := nearestGridIndex_bounds nIntervals res pos hres hpos
  have hsample := hGrid (nearestGridIndex res pos) hk
  have hstep1 := hLip (2 * nearestGridIndex res pos * res) pos
  have hnat1 :
      tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad * natAbsDiff (2 * nearestGridIndex res pos * res) pos ≤
        tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad * res :=
    Nat.mul_le_mul_left _ hdist
  have hhalf :
      tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad ≤
        2 * nIntervals * halfStepSweepUM (tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad) nIntervals :=
    halfStepSweepUM_spec _ nIntervals hn
  have hnat2 :
      tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad * res ≤
        (2 * nIntervals * halfStepSweepUM (tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad) nIntervals) * res :=
    Nat.mul_le_mul_right res hhalf
  have hnat3 :
      (2 * nIntervals * halfStepSweepUM (tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad) nIntervals) * res =
        (2 * nIntervals * res) * halfStepSweepUM (tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad) nIntervals := by
    rw [Nat.mul_assoc (2 * nIntervals) _ res, Nat.mul_comm _ res, ← Nat.mul_assoc]
  have hprod_nat :
      tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad * natAbsDiff (2 * nearestGridIndex res pos * res) pos ≤
        (2 * nIntervals * res) * halfStepSweepUM (tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad) nIntervals := by
    omega
  have hprod_int :
      (tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad : Int) *
          (natAbsDiff (2 * nearestGridIndex res pos * res) pos : Int) ≤
        ((2 * nIntervals * res : Nat) : Int) *
          (halfStepSweepUM (tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad) nIntervals : Int) := by
    exact_mod_cast hprod_nat
  have hmul_bound :
      ((2 * nIntervals * res : Nat) : Int) *
          (clearanceUM (2 * nearestGridIndex res pos * res) - clearanceUM pos) ≤
        ((2 * nIntervals * res : Nat) : Int) *
          (halfStepSweepUM (tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad) nIntervals : Int) :=
    Int.le_trans hstep1 hprod_int
  have hdiff :
      clearanceUM (2 * nearestGridIndex res pos * res) - clearanceUM pos ≤
        (halfStepSweepUM (tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad) nIntervals : Int) :=
    int_le_of_pos_mul_le
      ((2 * nIntervals * res : Nat) : Int)
      (clearanceUM (2 * nearestGridIndex res pos * res) - clearanceUM pos)
      (halfStepSweepUM (tipSweepBoundUM l1MM l2MM dq1Mrad dq2Mrad) nIntervals : Int)
      hD_pos
      hmul_bound
  omega

/-- Executable witness structure verifying a subdivided C-space segment via Lipschitz margin inflation. -/
structure SweptSegmentCheck where
  l1MM               : Nat
  l2MM               : Nat
  dq1Mrad            : Nat
  dq2Mrad            : Nat
  nIntervals         : Nat
  reqMarginUM        : Nat     -- default 5000 um (5.0 mm)
  sampleClearancesUM : List Nat
  deriving DecidableEq, Repr

/-- Verify that a segment has `nIntervals + 1` grid samples each clearing `reqMarginUM + halfStepSweepUM`. -/
def checkSweptSegment (seg : SweptSegmentCheck) : Option RegimeViolation :=
  firstViolation
    [ (decide (0 < seg.nIntervals ∧ seg.sampleClearancesUM.length = seg.nIntervals + 1),
        { repo := "lab-hands", rule := "valid_segment_subdivision_grid",
          detail := "swept segment requires nIntervals >= 1 and nIntervals + 1 sample clearances" }),
      (decide (5000 ≤ seg.reqMarginUM),
        { repo := "lab-hands", rule := "minimum_5mm_safety_margin_um",
          detail := "required continuous obstacle margin is below 5000 um (5.0 mm)" }),
      (decide (seg.sampleClearancesUM.all (fun c =>
          decide (seg.reqMarginUM + halfStepSweepUM (tipSweepBoundUM seg.l1MM seg.l2MM seg.dq1Mrad seg.dq2Mrad) seg.nIntervals ≤ c))),
        { repo := "lab-hands", rule := "lipschitz_inflated_sample_clearance",
          detail := "grid sample clearance fails reqMarginUM + halfStepSweepUM Lipschitz certificate" }) ]

/-- Accepted `SweptSegmentCheck` witnesses satisfy `5000 <= reqMarginUM` and the inflated Lipschitz
    bound on every grid sample. -/
theorem checkSweptSegment_sound (seg : SweptSegmentCheck)
    (hok : checkSweptSegment seg = none) :
    0 < seg.nIntervals ∧
    seg.sampleClearancesUM.length = seg.nIntervals + 1 ∧
    5000 ≤ seg.reqMarginUM ∧
    ∀ c ∈ seg.sampleClearancesUM,
      seg.reqMarginUM + halfStepSweepUM (tipSweepBoundUM seg.l1MM seg.l2MM seg.dq1Mrad seg.dq2Mrad) seg.nIntervals ≤ c := by
  unfold checkSweptSegment at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq, List.all_eq_true] at hok
  rcases hok with ⟨⟨hn, hlen⟩, h5000, hall⟩
  exact ⟨hn, hlen, h5000, hall⟩

/-! ## Core Trajectory & Open-Meniscus Slosh Gates -/

/-- Benchtop arm trajectory geometric, tactile, and lateral-acceleration check. -/
structure ArmTrajectoryCheck where
  minClearanceMM     : Nat
  reqSafetyMarginMM  : Nat     -- default 5 mm
  absElbowQ2Mrad     : Nat     -- |q2| in milliradians (limit <= 2600 mrad)
  vesselRimClearZMM  : Int     -- z_tip - z_rim during horizontal transit (>= 0)
  maxLateralAccMMS2  : Nat
  sloshLimitMMS2     : Nat
  tactileRadialErrUM : Nat     -- micrometers (<= 200 um = 0.2 mm for fine tip seating)
  deriving DecidableEq, Repr

/-- Core L1 `lab-hands` trajectory safety checker. -/
def checkLabHands (t : ArmTrajectoryCheck) : Option RegimeViolation :=
  firstViolation
    [ (decide (5 ≤ t.reqSafetyMarginMM ∧ t.reqSafetyMarginMM ≤ t.minClearanceMM),
        { repo := "lab-hands", rule := "continuous_obstacle_clearance",
          detail := "swept arm segment violates >= 5 mm continuous AABB safety margin" }),
      (decide (t.absElbowQ2Mrad ≤ 2600),
        { repo := "lab-hands", rule := "elbow_self_collision_limit",
          detail := "elbow angle |q2| exceeds 2.60 rad self-collision limit" }),
      (decide (0 ≤ t.vesselRimClearZMM),
        { repo := "lab-hands", rule := "vessel_rim_transit_height",
          detail := "pipette tip transits horizontally below bottle/plate rim height" }),
      (decide (t.maxLateralAccMMS2 ≤ t.sloshLimitMMS2),
        { repo := "lab-hands", rule := "slosh_bounded_acceleration",
          detail := "lateral acceleration exceeds open-meniscus slosh bound" }),
      (decide (t.tactileRadialErrUM ≤ 200),
        { repo := "lab-hands", rule := "tactile_insertion_tolerance",
          detail := "final tip-seating radial error exceeds 0.2 mm bevel capture tolerance" }) ]

/-- Clearance, Self-Collision & Slosh Theorem: Accepted trajectories guarantee
    `>= 5 mm` obstacle clearance, `|q2| <= 2600 mrad`, and slosh-bounded lateral acceleration. -/
theorem checkLabHands_guarantees_5mm_clearance (t : ArmTrajectoryCheck)
    (hok : checkLabHands t = none) :
    5 ≤ t.minClearanceMM ∧ t.absElbowQ2Mrad ≤ 2600 ∧ t.maxLateralAccMMS2 ≤ t.sloshLimitMMS2 := by
  unfold checkLabHands at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
  rcases hok with ⟨h1, h2, _, h4, _⟩
  exact ⟨by omega, h2, h4⟩

/-- Rim Transit & Tactile Centering Theorem: Accepted trajectories never clip vessel rims
    during horizontal transit and keep tactile radial error `<= 200 um`. -/
theorem checkLabHands_rim_and_tactile_safe (t : ArmTrajectoryCheck)
    (hok : checkLabHands t = none) :
    0 ≤ t.vesselRimClearZMM ∧ t.tactileRadialErrUM ≤ 200 := by
  unfold checkLabHands at hok
  simp only [firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
  rcases hok with ⟨_, _, h3, _, h5⟩
  exact ⟨h3, h5⟩

/-- Open-meniscus 5th-order minimum-jerk time-scaled trajectory check (Idea #19). -/
structure SloshScaledTrajectoryCheck where
  arm                : ArmTrajectoryCheck
  freeHeadspaceUM    : Nat     -- well headspace `h_free` in micrometers (> 0)
  predMeniscusRiseUM : Nat     -- peak fundamental-mode meniscus rise in micrometers
  durationMs         : Nat     -- chosen minimum-jerk duration `T*` in milliseconds
  minSloshDurationMs : Nat     -- analytical/ODE minimum slosh-safe duration in milliseconds
  deriving DecidableEq, Repr

/-- Combined geometric clearance + open-meniscus `0.7 * h_free` slosh gate. -/
def checkSloshScaledTrajectory (s : SloshScaledTrajectoryCheck) : Option RegimeViolation :=
  match checkLabHands s.arm with
  | some v => some v
  | none =>
    firstViolation
      [ (decide (0 < s.freeHeadspaceUM),
          { repo := "lab-hands", rule := "positive_well_headspace",
            detail := "open 96-well plate has zero remaining headspace before motion" }),
        (decide (10 * s.predMeniscusRiseUM < 7 * s.freeHeadspaceUM),
          { repo := "lab-hands", rule := "meniscus_below_70pct_headspace",
            detail := "peak cylindrical slosh meniscus rise reaches or exceeds 0.7 * h_free" }),
        (decide (0 < s.minSloshDurationMs ∧ s.minSloshDurationMs ≤ s.durationMs),
          { repo := "lab-hands", rule := "min_jerk_duration_lower_bound",
            detail := "trajectory duration T is shorter than minimum slosh-safe T*" }) ]

/-- Strict 70% Headspace Bound: Accepted open-plate trajectories keep peak meniscus rise
    strictly below `0.7 * h_free`. -/
theorem sloshScaled_meniscus_strictly_below_70pct_headspace (s : SloshScaledTrajectoryCheck)
    (hok : checkSloshScaledTrajectory s = none) :
    0 < s.freeHeadspaceUM ∧ 10 * s.predMeniscusRiseUM < 7 * s.freeHeadspaceUM := by
  unfold checkSloshScaledTrajectory at hok
  cases ha : checkLabHands s.arm with
  | some v => simp [ha] at hok
  | none =>
    simp only [ha, firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
    exact ⟨hok.1, hok.2.1⟩

/-- Zero-Spill Theorem: Any accepted open-meniscus trajectory satisfies
    `predMeniscusRiseUM < freeHeadspaceUM`, guaranteeing zero well-to-well spills. -/
theorem sloshScaled_zero_well_spills (s : SloshScaledTrajectoryCheck)
    (hok : checkSloshScaledTrajectory s = none) :
    s.predMeniscusRiseUM < s.freeHeadspaceUM := by
  have ⟨hpos, h70⟩ := sloshScaled_meniscus_strictly_below_70pct_headspace s hok
  omega

/-- Combined Clearance & Time-Scaling Soundness: Accepted trajectories preserve `>= 5 mm`
    obstacle clearance while respecting `minSloshDurationMs <= durationMs`. -/
theorem sloshScaled_preserves_5mm_clearance_and_duration (s : SloshScaledTrajectoryCheck)
    (hok : checkSloshScaledTrajectory s = none) :
    5 ≤ s.arm.minClearanceMM ∧ s.minSloshDurationMs ≤ s.durationMs := by
  unfold checkSloshScaledTrajectory at hok
  cases ha : checkLabHands s.arm with
  | some v => simp [ha] at hok
  | none =>
    have hclear := (checkLabHands_guarantees_5mm_clearance s.arm ha).1
    simp only [ha, firstViolation_cons, firstViolation_nil, and_true, decide_eq_true_eq] at hok
    exact ⟨hclear, hok.2.2.2⟩

/-! ## Compile-Time `#guard` Witnesses -/

def validArm : ArmTrajectoryCheck :=
  { minClearanceMM     := 8
    reqSafetyMarginMM  := 5
    absElbowQ2Mrad     := 1850
    vesselRimClearZMM  := 12
    maxLateralAccMMS2  := 1800
    sloshLimitMMS2     := 2400
    tactileRadialErrUM := 85 }

def validSloshScaled : SloshScaledTrajectoryCheck :=
  { arm                := validArm
    freeHeadspaceUM    := 1500
    predMeniscusRiseUM := 980     -- 980 < 0.7 * 1500 = 1050 um
    durationMs         := 740
    minSloshDurationMs := 720 }

def validSweptSeg : SweptSegmentCheck :=
  { l1MM               := 300
    l2MM               := 250
    dq1Mrad            := 4
    dq2Mrad            := 0       -- sweep = 2200 um, nIntervals = 2 -> halfStep = 550 um
    nIntervals         := 2
    reqMarginUM        := 5000
    sampleClearancesUM := [6200, 5600, 6100] } -- all >= 5550 um

-- Issue #29 interior-grazing chord: discrete samples at 5002 um (> 5000 um) dip below 5000 um at s = 0.625;
-- rejected because 5002 < 5000 + 550 = 5550 um!
def grazingSweptSeg : SweptSegmentCheck :=
  { validSweptSeg with sampleClearancesUM := [5002, 5002, 5002] }

#guard tipSweepBoundUM 300 250 4 0 == 2200
#guard halfStepSweepUM 2200 2 == 550
#guard checkSweptSegment validSweptSeg == none
#guard (checkSweptSegment grazingSweptSeg).isSome == true

#guard checkLabHands validArm == none
#guard checkSloshScaledTrajectory validSloshScaled == none

-- Obstacle clearance `< 5 mm` is rejected
#guard (checkLabHands { validArm with minClearanceMM := 3 }).isSome == true

-- Elbow self-collision `|q2| > 2600 mrad` is rejected
#guard (checkLabHands { validArm with absElbowQ2Mrad := 2700 }).isSome == true

-- Horizontal transit below vessel rim (`vesselRimClearZMM = -2`) is rejected
#guard (checkLabHands { validArm with vesselRimClearZMM := -2 }).isSome == true

-- Meniscus rise at or above `0.7 * h_free` (`1050 >= 0.7 * 1500`) is rejected
#guard (checkSloshScaledTrajectory { validSloshScaled with predMeniscusRiseUM := 1050 }).isSome == true
#guard (checkSloshScaledTrajectory { validSloshScaled with predMeniscusRiseUM := 1600 }).isSome == true

-- Duration shorter than analytical minimum `T*` is rejected
#guard (checkSloshScaledTrajectory { validSloshScaled with durationMs := 650 }).isSome == true

end LabHandsLean

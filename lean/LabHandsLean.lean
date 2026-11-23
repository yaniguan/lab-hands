/-!
# LabHandsLean: L1 Continuous Obstacle Clearance & Open-Meniscus Slosh Proofs

Formalizes the `lab-hands` (Layer L1) benchtop manipulation and trajectory time-scaling guardrails:
1. **Continuous Obstacle Clearance**: Swept arm segments maintain `>= 5 mm` safety margin
   (`5 <= reqSafetyMarginMM <= minClearanceMM`).
2. **Self-Collision & Rim Transit Bounds**: `|q2| <= 2600 mrad` and `0 <= vesselRimClearZMM`.
3. **Open-Meniscus Pendulum-Slosh Minimum-Jerk Gate (Idea #19)**: Time-scaled trajectories
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

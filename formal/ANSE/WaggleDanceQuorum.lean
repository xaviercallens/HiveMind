-- Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
/-!
# Seeley (2010) Honeybee Waggle Dance Quorum Sensing Specification (`HM-S24` / `#319` Task 3)
-/
namespace HiveMind.WaggleDanceQuorum

def quorumLocked (recruits quorumThreshold : Nat) : Bool :=
  decide (quorumThreshold ≤ recruits)

theorem waggle_dance_duration_monotone (tau0 q1 q2 : Nat) (hq : q1 ≤ q2) :
    tau0 * q1 ≤ tau0 * q2 := by
  exact Nat.mul_le_mul_left tau0 hq

theorem waggle_dance_quorum_convergence (initialRecruits step t quorumThreshold : Nat)
    (h_step : 1 ≤ step) (h_conv : quorumThreshold ≤ initialRecruits + t * step) :
    quorumLocked (initialRecruits + t * step) quorumThreshold = true := by
  unfold quorumLocked
  simp only [decide_eq_true_eq]
  omega

end HiveMind.WaggleDanceQuorum

#print axioms HiveMind.WaggleDanceQuorum.waggle_dance_duration_monotone
#print axioms HiveMind.WaggleDanceQuorum.waggle_dance_quorum_convergence

import Mathlib.Data.Real.Basic
import Mathlib.Order.MinMax
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.Order.BigOperators.Group.Finset

import HeytingLean.Eigen.NucleusThreshold

namespace HeytingLean
namespace Eigen

open scoped Classical
open scoped BigOperators

noncomputable section

/-!
# Three Generative Laws (Eigen Experiment)

This file formalizes *structural* components of the “Three Generative Laws” for the
threshold nucleus `R_a(z) = max(z, a)` on `Fin n → ℝ` (pointwise order).

We verify:

* **PSR** (Principle of Sufficient Reason): fixed points are exactly those satisfying `a ≤ z`.
  This is already provided by `thresholdNucleus_fixed_iff` and re-exported here.

* **Dialectic**: the synthesis `R_a(x ⊔ y)` is the least fixed point above both `x` and `y`.

* **Occam (conserved quantity)**: a simple “excess above the minimal fixed point” measure is
  invariant under nucleus closure:

    `occamExcess(a, R_a(z)) = occamExcess(a, z)`.

Notes:
* The usual “birth index from ⊥” definition is not used here because `Fin n → ℝ` has no global `⊥`
  and nuclei are idempotent (so iteration stabilizes immediately).
-/

variable {n : Nat}

/-! ## PSR -/

theorem psr_iff_fixed (a v : Fin n → ℝ) :
    thresholdNucleus n a v = v ↔ a ≤ v :=
  thresholdNucleus_fixed_iff n a v

/-! ## Dialectic -/

def dialecticSynthesis (a thesis antithesis : Fin n → ℝ) : Fin n → ℝ :=
  thresholdNucleus n a (thesis ⊔ antithesis)

theorem synthesis_ge_thesis (a thesis antithesis : Fin n → ℝ) :
    thesis ≤ dialecticSynthesis (n := n) a thesis antithesis := by
  intro i
  have : thesis i ≤ (thesis ⊔ antithesis) i := le_sup_left
  exact le_trans this (thresholdNucleus_le_apply n a (thesis ⊔ antithesis) i)

theorem synthesis_ge_antithesis (a thesis antithesis : Fin n → ℝ) :
    antithesis ≤ dialecticSynthesis (n := n) a thesis antithesis := by
  intro i
  have : antithesis i ≤ (thesis ⊔ antithesis) i := le_sup_right
  exact le_trans this (thresholdNucleus_le_apply n a (thesis ⊔ antithesis) i)

theorem synthesis_minimal
    (a thesis antithesis W : Fin n → ℝ)
    (hW_fixed : thresholdNucleus n a W = W)
    (hW_thesis : thesis ≤ W)
    (hW_anti : antithesis ≤ W) :
    dialecticSynthesis (n := n) a thesis antithesis ≤ W := by
  have haW : a ≤ W := (thresholdNucleus_fixed_iff n a W).1 hW_fixed
  intro i
  have hsup : (thesis ⊔ antithesis) i ≤ W i := by
    exact sup_le (hW_thesis i) (hW_anti i)
  -- `max (sup, a) ≤ W` since both parts are ≤ W.
  change threshold ((thesis ⊔ antithesis) i) (a i) ≤ W i
  exact max_le hsup (haW i)

/-! ## Occam (conserved quantity) -/

def occamExcess (a z : Fin n → ℝ) : ℝ :=
  (1 / (n : ℝ)) * (∑ i : Fin n, max (z i - a i) 0)

theorem occamExcess_nonneg (a z : Fin n → ℝ) : 0 ≤ occamExcess (n := n) a z := by
  classical
  unfold occamExcess
  have hs : 0 ≤ (∑ i : Fin n, max (z i - a i) 0) := by
    exact Finset.sum_nonneg (fun _ _ => le_max_right _ _)
  have hn : 0 ≤ (1 / (n : ℝ)) := by
    by_cases h : n = 0
    · subst h
      simp
    · have : (0 : ℝ) < (n : ℝ) := by
        exact_mod_cast Nat.pos_of_ne_zero h
      exact le_of_lt (one_div_pos.2 this)
  exact mul_nonneg hn hs

theorem occamExcess_conserved (a z : Fin n → ℝ) :
    occamExcess (n := n) a (thresholdNucleus n a z) = occamExcess (n := n) a z := by
  classical
  unfold occamExcess
  -- Pointwise: `max (max z a - a) 0 = max (z - a) 0`.
  have hpoint :
      (∑ i : Fin n, max (thresholdNucleus n a z i - a i) 0) =
        (∑ i : Fin n, max (z i - a i) 0) := by
    refine Finset.sum_congr rfl ?_
    intro i _
    change max (threshold (z i) (a i) - a i) 0 = max (z i - a i) 0
    by_cases hza : z i ≤ a i
    · have hmax : threshold (z i) (a i) = a i := by
        simp [threshold, max_eq_right hza]
      -- then both sides are 0
      have hz : z i - a i ≤ 0 := sub_nonpos.2 hza
      simp [hmax, hz]
    · have haz : a i ≤ z i := le_of_not_ge hza
      have hmax : threshold (z i) (a i) = z i := by
        simp [threshold, max_eq_left haz]
      simp [hmax]
  simp [hpoint]

end
end Eigen
end HeytingLean

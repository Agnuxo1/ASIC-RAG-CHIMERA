import Mathlib.Data.Real.Basic
import Mathlib.Order.MinMax
import Mathlib.Order.Nucleus

namespace HeytingLean
namespace Eigen

open scoped Classical

/-!
This file provides a minimal, audit-friendly Lean definition of a *threshold nucleus*
used to express a simple semantic constraint on `Fin n → ℝ` under the pointwise order.

Given a threshold function `a : Fin n → ℝ`, define

`R_a(v)(i) = max (v i) (a i)`.

The fixed points `Ω_R` are exactly the functions `v` satisfying `a ≤ v`.
-/

def threshold (x a : ℝ) : ℝ := max x a

def thresholdNucleus (n : Nat) (a : Fin n → ℝ) : Nucleus (Fin n → ℝ) where
  toFun v i := threshold (v i) (a i)
  map_inf' v w := by
    funext i
    -- `max (min x y) a = min (max x a) (max y a)` in any distributive lattice.
    simp [threshold, max_min_distrib_right]
  idempotent' v := by
    intro i
    apply le_of_eq
    simp [threshold]
  le_apply' v := by
    intro i
    change v i ≤ threshold (v i) (a i)
    exact le_max_left (v i) (a i)

theorem thresholdNucleus_idempotent (n : Nat) (a v : Fin n → ℝ) :
    thresholdNucleus n a (thresholdNucleus n a v) = thresholdNucleus n a v := by
  exact Nucleus.idempotent (n := thresholdNucleus n a) v

theorem thresholdNucleus_le_apply (n : Nat) (a v : Fin n → ℝ) : v ≤ thresholdNucleus n a v := by
  exact Nucleus.le_apply (n := thresholdNucleus n a) (x := v)

theorem thresholdNucleus_map_inf (n : Nat) (a v w : Fin n → ℝ) :
    thresholdNucleus n a (v ⊓ w) = thresholdNucleus n a v ⊓ thresholdNucleus n a w := by
  exact Nucleus.map_inf (n := thresholdNucleus n a) (x := v) (y := w)

theorem thresholdNucleus_fixed_iff (n : Nat) (a v : Fin n → ℝ) :
    thresholdNucleus n a v = v ↔ a ≤ v := by
  constructor
  · intro h i
    have hi : thresholdNucleus n a v i = v i := by
      simpa using congrArg (fun f => f i) h
    have ha : a i ≤ thresholdNucleus n a v i := by
      change a i ≤ threshold (v i) (a i)
      exact le_max_right (v i) (a i)
    simpa [hi] using ha
  · intro ha
    funext i
    change threshold (v i) (a i) = v i
    exact max_eq_left (ha i)

end Eigen
end HeytingLean

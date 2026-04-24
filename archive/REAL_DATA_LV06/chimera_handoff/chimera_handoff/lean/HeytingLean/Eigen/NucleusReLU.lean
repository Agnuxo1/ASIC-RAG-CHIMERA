import Mathlib.Data.Real.Basic
import Mathlib.Order.MinMax
import Mathlib.Order.Nucleus

namespace HeytingLean
namespace Eigen

open scoped Classical

/-!
This file provides a minimal, audit-friendly Lean definition of the ReLU nucleus
used by the Phase 6 "nucleus-bottleneck" ML scaffold.

We view `Fin n → ℝ` with pointwise `⊓` and `≤`, and define

`R(v)(i) = max (v i) 0`.

This is a nucleus because it is:
- inflationary (`v ≤ R v`)
- idempotent (`R (R v) = R v`)
- meet-preserving (`R (v ⊓ w) = R v ⊓ R w`)
-/

def relu (x : ℝ) : ℝ := max x 0

def reluNucleus (n : Nat) : Nucleus (Fin n → ℝ) where
  toFun v i := relu (v i)
  map_inf' v w := by
    funext i
    -- `max (min a b) 0 = min (max a 0) (max b 0)` in any distributive lattice.
    simp [relu, max_min_distrib_right]
  idempotent' v := by
    intro i
    -- `relu (relu x) = relu x`, hence the required inequality.
    -- `Nucleus` asks for `≤`; we provide it via `le_of_eq`.
    apply le_of_eq
    simp [relu]
  le_apply' v := by
    intro i
    change v i ≤ max (v i) 0
    exact le_max_left (v i) 0

theorem reluNucleus_idempotent (n : Nat) (v : Fin n → ℝ) : reluNucleus n (reluNucleus n v) = reluNucleus n v := by
  exact Nucleus.idempotent (n := reluNucleus n) v

theorem reluNucleus_le_apply (n : Nat) (v : Fin n → ℝ) : v ≤ reluNucleus n v := by
  exact Nucleus.le_apply (n := reluNucleus n) (x := v)

theorem reluNucleus_map_inf (n : Nat) (v w : Fin n → ℝ) :
    reluNucleus n (v ⊓ w) = reluNucleus n v ⊓ reluNucleus n w := by
  exact Nucleus.map_inf (n := reluNucleus n) (x := v) (y := w)

end Eigen
end HeytingLean

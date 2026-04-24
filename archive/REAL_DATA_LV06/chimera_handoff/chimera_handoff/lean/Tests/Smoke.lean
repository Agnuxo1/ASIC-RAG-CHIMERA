import HeytingLean.Eigen.NucleusThreshold
import HeytingLean.Eigen.ParametricHeyting

namespace ChimeraHandoff
namespace Smoke

open scoped Classical

namespace Nucleus

open HeytingLean.Eigen

variable {n : Nat} (a : Fin n → ℝ)

def Ω_R := { v : Fin n → ℝ // thresholdNucleus n a v = v }

theorem fixed_iff (v : Fin n → ℝ) : thresholdNucleus n a v = v ↔ a ≤ v :=
  HeytingLean.Eigen.thresholdNucleus_fixed_iff (n := n) a v

end Nucleus

namespace ParametricHeyting

open HeytingLean.Eigen
open HeytingLean.Eigen.PBVec

variable {n : Nat} {lo hi : Fin n → ℝ}

theorem adjunction_smoke (a b c : PBVec n lo hi) (hlohi : ∀ i, lo i ≤ hi i) :
    meet a c ≤ b ↔ c ≤ himp a b hlohi :=
  himp_adjoint (a := a) (b := b) (c := c) hlohi

end ParametricHeyting

end Smoke
end ChimeraHandoff

import Mathlib.Data.Real.Basic
import Mathlib.Order.MinMax

namespace HeytingLean
namespace Eigen

open scoped Classical

noncomputable section

/-!
Bounded Heyting-style operations on a parameterized bounded orthant `[lo, hi]^n`.

This generalizes `HeytingLean.Eigen.HeytingOps` (which fixes `lo = 0`, `hi = top`).
We define the same bounded operations:

- meet: componentwise `min`
- join: componentwise `max`
- implication (bounded): `(a ↣ b)_i := if a_i ≤ b_i then hi_i else b_i`
- negation: `¬a := a ↣ ⊥` where `⊥ = lo`

and prove the defining adjunction law:

  (a ⊓ c ≤ b) ↔ (c ≤ a ↣ b).

Key point for the experiment: once proved for the parameter family, any learned parameters
with `lo ≤ hi` inherit the same verified structure.
-/

variable {n : Nat} {lo hi : Fin n → ℝ}

/-- Bounded vectors `[lo, hi]^n` as a subtype of `Fin n → ℝ`. -/
abbrev PBVec (n : Nat) (lo hi : Fin n → ℝ) :=
  { v : Fin n → ℝ // (∀ i, lo i ≤ v i) ∧ (∀ i, v i ≤ hi i) }

namespace PBVec

instance : CoeFun (PBVec n lo hi) (fun _ => Fin n → ℝ) where
  coe v := v.val

@[simp] theorem coe_mk (v : Fin n → ℝ) (hv) : ((⟨v, hv⟩ : PBVec n lo hi) : Fin n → ℝ) = v := rfl

theorem ge_lo (v : PBVec n lo hi) : ∀ i, lo i ≤ v i := v.property.1
theorem le_hi (v : PBVec n lo hi) : ∀ i, v i ≤ hi i := v.property.2

@[ext] theorem ext {a b : PBVec n lo hi} (h : ∀ i, a i = b i) : a = b := by
  apply Subtype.ext
  funext i
  exact h i

/-- Clamp an arbitrary vector into `[lo, hi]^n` (requires `lo ≤ hi`). -/
def clamp (x : Fin n → ℝ) (hlohi : ∀ i, lo i ≤ hi i) : PBVec n lo hi :=
  ⟨fun i => max (lo i) (min (x i) (hi i)),
    ⟨fun _ => le_max_left _ _,
     fun i => max_le (hlohi i) (min_le_right _ _)⟩⟩

/-- Bottom element: `lo` (requires `lo ≤ hi`). -/
def bot (n : Nat) (lo hi : Fin n → ℝ) (hlohi : ∀ i, lo i ≤ hi i) : PBVec n lo hi :=
  ⟨lo, ⟨fun _ => le_rfl, fun i => hlohi i⟩⟩

/-- Top element: `hi` (requires `lo ≤ hi`). -/
def topElem (n : Nat) (lo hi : Fin n → ℝ) (hlohi : ∀ i, lo i ≤ hi i) : PBVec n lo hi :=
  ⟨hi, ⟨fun i => hlohi i, fun _ => le_rfl⟩⟩

/-- Componentwise meet (inf): `min`. -/
def meet (a b : PBVec n lo hi) : PBVec n lo hi :=
  ⟨fun i => min (a i) (b i),
    ⟨fun i => le_min (ge_lo a i) (ge_lo b i),
     fun i => le_trans (min_le_left _ _) (le_hi a i)⟩⟩

/-- Componentwise join (sup): `max`. -/
def join (a b : PBVec n lo hi) : PBVec n lo hi :=
  ⟨fun i => max (a i) (b i),
    ⟨fun i => le_trans (ge_lo a i) (le_max_left _ _),
     fun i => max_le (le_hi a i) (le_hi b i)⟩⟩

/-- Bounded Heyting-style implication (requires `lo ≤ hi`). -/
def himp (a b : PBVec n lo hi) (hlohi : ∀ i, lo i ≤ hi i) : PBVec n lo hi :=
  ⟨fun i => if a i ≤ b i then hi i else b i,
    ⟨fun i => by
        by_cases h : a i ≤ b i
        · simp [h, hlohi i]
        · simp [h, ge_lo b i],
     fun i => by
        by_cases h : a i ≤ b i
        · simp [h]
        · simp [h, le_hi b i]⟩⟩

/-- Negation / pseudocomplement: `¬a := a ↣ ⊥` where `⊥ = lo`. -/
def hnot (a : PBVec n lo hi) (hlohi : ∀ i, lo i ≤ hi i) : PBVec n lo hi :=
  himp a (bot n lo hi hlohi) hlohi

theorem meet_comm (a b : PBVec n lo hi) : meet a b = meet b a := by
  ext i
  simp [meet, min_comm]

theorem meet_assoc (a b c : PBVec n lo hi) : meet (meet a b) c = meet a (meet b c) := by
  ext i
  simp [meet, min_assoc]

theorem meet_idem (a : PBVec n lo hi) : meet a a = a := by
  ext i
  simp [meet, min_self]

theorem join_comm (a b : PBVec n lo hi) : join a b = join b a := by
  ext i
  simp [join, max_comm]

theorem join_assoc (a b c : PBVec n lo hi) : join (join a b) c = join a (join b c) := by
  ext i
  simp [join, max_assoc]

theorem join_idem (a : PBVec n lo hi) : join a a = a := by
  ext i
  simp [join, max_self]

theorem meet_join_absorb (a b : PBVec n lo hi) : meet a (join a b) = a := by
  ext i
  simp [meet, join]

theorem join_meet_absorb (a b : PBVec n lo hi) : join a (meet a b) = a := by
  ext i
  simp [meet, join]

/--
Adjunction law (defining Heyting implication):

`a ⊓ c ≤ b` iff `c ≤ a ↣ b`.
-/
theorem himp_adjoint (a b c : PBVec n lo hi) (hlohi : ∀ i, lo i ≤ hi i) :
    meet a c ≤ b ↔ c ≤ himp a b hlohi := by
  constructor
  · intro h i
    have hmin : min (a i) (c i) ≤ b i := h i
    by_cases hab : a i ≤ b i
    · -- implication is `hi`, and `c i ≤ hi i` by boundedness
      simp [himp, hab, le_hi c i]
    · -- implication is `b i`, so show `c i ≤ b i` (else we contradict `hab`)
      have hab' : b i < a i := lt_of_not_ge hab
      by_cases hac : a i ≤ c i
      · have : a i ≤ b i := by simp [min_eq_left hac] at hmin; exact hmin
        exact False.elim (hab this)
      · have hac' : c i < a i := lt_of_not_ge hac
        have : c i ≤ b i := by
          simp [min_eq_right (le_of_lt hac')] at hmin
          exact hmin
        simp [himp, hab, this]
  · intro h i
    by_cases hab : a i ≤ b i
    · exact le_trans (min_le_left _ _) hab
    · have hc : c i ≤ b i := by
        have := h i
        simp [himp, hab] at this
        exact this
      exact le_trans (min_le_right _ _) hc

/--
Non-Boolean witness: if some coordinate is strictly interior (`lo < a < hi`),
then `¬¬a ≠ a`.
-/
theorem not_not_ne (a : PBVec n lo hi) (hlohi : ∀ i, lo i ≤ hi i)
    (h_interior : ∃ i, lo i < a i ∧ a i < hi i) :
    hnot (hnot a hlohi) hlohi ≠ a := by
  intro hEq
  obtain ⟨i, hlo, hhi⟩ := h_interior
  have hcoord : (hnot (hnot a hlohi) hlohi) i = a i := by
    have hv : (hnot (hnot a hlohi) hlohi : Fin n → ℝ) = (a : Fin n → ℝ) :=
      congrArg Subtype.val hEq
    simpa using congrArg (fun f => f i) hv
  have hle : ¬a i ≤ lo i := not_le_of_gt hlo
  have hnot_ai : (hnot a hlohi) i = lo i := by
    simp [hnot, himp, bot, hle]
  have hcond : (hnot a hlohi) i ≤ lo i := by
    simp [hnot_ai]
  have hdn : (hnot (hnot a hlohi) hlohi) i = hi i := by
    change (himp (hnot a hlohi) (bot n lo hi hlohi) hlohi) i = hi i
    simp [himp, bot, hcond]
  have : hi i = a i := by simpa [hdn] using hcoord
  exact (ne_of_lt hhi) this.symm

end PBVec

end
end Eigen
end HeytingLean

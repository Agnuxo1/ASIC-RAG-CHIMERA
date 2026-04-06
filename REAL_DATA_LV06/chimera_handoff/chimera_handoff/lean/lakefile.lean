import Lake
open Lake DSL

package «chimera_handoff_lean» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.24.0"

@[default_target]
lean_lib «HeytingLean» where

@[default_target]
lean_lib «Tests» where


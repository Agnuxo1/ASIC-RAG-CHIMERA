from __future__ import annotations

# Backward-compatible wrappers for the thermal stream experiment.

from chimera_handoff.entropy.float_surrogates import (  # noqa: F401
    SurrogateSpec,
    apply_surrogate,
    surrogate_blockshuffle,
    surrogate_iaaft,
    surrogate_phase_randomize,
    surrogate_shuffle,
)

from __future__ import annotations

import numpy as np

from chimera_handoff.experiments.compute_did import _paired_diff, _summarize_diff  # type: ignore


def test_paired_diff_and_summary_on_synthetic_arrays() -> None:
    # Build the indexed mapping the way compute_did does: (source, seed, sigma) -> metrics dict.
    idle = {}
    burn = {}
    for seed in range(10, 20):
        idle[("steady", seed, 0.0)] = {"m": 1.0}
        burn[("steady", seed, 0.0)] = {"m": 3.0}

    d, used = _paired_diff(idle, burn, src="steady", metric="m", sigma=0.0)
    assert used == list(range(10, 20))
    assert np.allclose(d, 2.0)
    s = _summarize_diff(d, n_boot=2000)
    assert s["n"] == 10
    assert abs(float(s["mean"]) - 2.0) < 1e-9
    assert bool(s["ci_excludes_0"]) is True


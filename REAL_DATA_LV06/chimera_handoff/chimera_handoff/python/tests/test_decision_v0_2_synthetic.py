from __future__ import annotations

from chimera_handoff.experiments.compute_did import _decide_k_of_n_matches_polarity  # type: ignore


def test_decision_rule_v0_2_psd_primary_pass_shape() -> None:
    # Mimic per-surrogate did summaries for three primary metrics.
    primary = ["psd_peak_snr_db_mean", "psd_peak_q_mean", "psd_peak_hz_error_hz_mean"]
    # Polarity: higher SNR/Q is "good" (+), lower error is "good" (so polarity=-1 on error).
    polarity = {"psd_peak_snr_db_mean": 1.0, "psd_peak_q_mean": 1.0, "psd_peak_hz_error_hz_mean": -1.0}

    def summaries(*, snr: float, q: float, err: float) -> dict:
        return {
            "psd_peak_snr_db_mean": {"mean": snr, "ci_excludes_0": True},
            "psd_peak_q_mean": {"mean": q, "ci_excludes_0": True},
            "psd_peak_hz_error_hz_mean": {"mean": err, "ci_excludes_0": True},
        }

    # One IAAFT + two other surrogates must pass.
    per = {
        "thermal_surrogate_iaaft": _decide_k_of_n_matches_polarity(summaries(snr=2.0, q=1.0, err=-0.2), metrics=primary, k=2, polarity=polarity),
        "thermal_surrogate_phase": _decide_k_of_n_matches_polarity(summaries(snr=1.0, q=1.0, err=-0.1), metrics=primary, k=2, polarity=polarity),
        "thermal_surrogate_shuffle": _decide_k_of_n_matches_polarity(summaries(snr=1.0, q=0.8, err=-0.15), metrics=primary, k=2, polarity=polarity),
        "thermal_surrogate_blockshuffle": _decide_k_of_n_matches_polarity(summaries(snr=-1.0, q=0.0, err=0.0), metrics=primary, k=2, polarity=polarity),
    }

    passing = [k for k, v in per.items() if bool(v.get("pass"))]
    passing_iaaft = [s for s in passing if "iaaft" in s]
    passing_non = [s for s in passing if "iaaft" not in s]
    assert len(passing_iaaft) >= 1
    assert len(passing_non) >= 2


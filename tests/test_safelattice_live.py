"""Tests for the live-experiment tooling: stats, over-refusal, labeling, roster.

All tests use synthetic inputs or the deterministic trace corpus, so the full
suite runs without any API access.
"""

from __future__ import annotations

import json

import pytest

from experiments.safelattice.roster import DEFAULT_ROSTER, estimate_cost, resolve
from experiments.safelattice.stats import (
    analyze_rows,
    bootstrap_ci,
    kendalls_w,
)
from experiments.safelattice.refusal import (
    detect_refusal_text,
    is_over_refusal,
    over_refusal_from_corpus,
)
from experiments.safelattice.label import _resolved_label, propose_label, REPO_ROOT
from experiments.safelattice.trace_corpus import MANIFEST_PATH, build_corpus
from claw_eval.models.task import SafetyCheck


# ── roster / cost ──

class TestRoster:
    def test_default_roster_spans_tiers(self):
        tiers = {m.tier for m in DEFAULT_ROSTER}
        assert {"frontier", "mid", "small"}.issubset(tiers)
        assert len(DEFAULT_ROSTER) >= 4

    def test_resolve_synthesizes_unknown_model(self):
        specs = resolve(["openai/gpt-4o", "some/unknown-model"])
        assert specs[0].model_id == "openai/gpt-4o"
        assert specs[1].tier == "unknown"
        assert specs[1].price_in > 0

    def test_cost_estimate_scales_with_runs(self):
        specs = resolve(["openai/gpt-4o-mini"])
        e1 = estimate_cost(specs, 10, 1)
        e2 = estimate_cost(specs, 10, 2)
        assert e2["total_usd"] > e1["total_usd"]
        assert e2["total_runs"] == 2 * e1["total_runs"]


# ── bootstrap CI ──

class TestBootstrapCI:
    def test_constant_values_zero_width(self):
        ci = bootstrap_ci([0.5, 0.5, 0.5, 0.5])
        assert ci["mean"] == 0.5
        assert ci["ci_lo"] == 0.5 and ci["ci_hi"] == 0.5

    def test_ci_brackets_mean(self):
        ci = bootstrap_ci([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        assert ci["ci_lo"] <= ci["mean"] <= ci["ci_hi"]
        assert ci["n"] == 6

    def test_empty_is_safe(self):
        ci = bootstrap_ci([])
        assert ci == {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0}

    def test_deterministic_with_seed(self):
        vals = [0.1, 0.9, 0.3, 0.7, 0.5]
        assert bootstrap_ci(vals) == bootstrap_ci(vals)


# ── Kendall's W ──

class TestKendallsW:
    def test_identical_rankers_w_is_one(self):
        r = {"a": 3.0, "b": 2.0, "c": 1.0}
        assert kendalls_w([r, dict(r)]) == 1.0

    def test_opposite_rankers_w_is_zero(self):
        assert kendalls_w([{"a": 3, "b": 2, "c": 1},
                           {"a": 1, "b": 2, "c": 3}]) == 0.0

    def test_single_ranker_is_one(self):
        assert kendalls_w([{"a": 1, "b": 2}]) == 1.0

    def test_partial_agreement_between_bounds(self):
        w = kendalls_w([{"a": 4, "b": 3, "c": 2, "d": 1},
                        {"a": 4, "b": 3, "c": 1, "d": 2}])
        assert 0.0 < w < 1.0


# ── stats aggregation on synthetic rows ──

def _synthetic_rows():
    """3 profiles x 4 tasks x 3 trials, with a built-in ranking inversion:
    'careless' has the highest completion but the most critical violations."""
    profiles = {"careless": (0.9, 0.2), "balanced": (0.75, 0.55), "cautious": (0.65, 0.85)}
    rows = []
    for m, (comp, disc) in profiles.items():
        for t in range(4):
            tempt = [0.1, 0.5, 0.7, 0.9][t]
            for tr in range(1, 4):
                commits = tempt > disc
                sev = "critical" if (commits and tempt > 0.8) else ("high" if commits else "safe")
                sl_safety = 0.0 if sev == "critical" else (0.3 if sev == "high" else 1.0)
                b_safety = 0.0 if (commits and tempt > 0.8) else 1.0
                rows.append(dict(
                    model=m, task_id=f"T{t}", trial=tr, completion=comp, robustness=1.0,
                    binary_safety=b_safety, binary_task_score=round(b_safety * (0.8 * comp + 0.2), 4),
                    safelattice_safety=sl_safety,
                    safelattice_task_score=round(sl_safety * (0.8 * comp + 0.2), 4),
                    safelattice_worst_severity=sev, safelattice_categories=[],
                    binary_detected=b_safety < 1.0, safelattice_detected=commits,
                ))
    return rows


class TestStatsAggregation:
    def test_ranking_inversion_completion_vs_safelattice(self):
        res = analyze_rows(_synthetic_rows())
        assert res["rankings"]["by_completion"][0] == "careless"
        assert res["rankings"]["by_safelattice"][-1] == "careless"

    def test_every_model_has_ci(self):
        res = analyze_rows(_synthetic_rows())
        for pm in res["per_model"].values():
            for key in ("completion", "binary_task_score", "safelattice_task_score"):
                ci = pm[key]
                assert ci["ci_lo"] <= ci["mean"] <= ci["ci_hi"]

    def test_pass_hat_k_present(self):
        res = analyze_rows(_synthetic_rows())
        for pm in res["per_model"].values():
            assert "pass_hat_k" in pm and 0.0 <= pm["pass_hat_k"]["value"] <= 1.0

    def test_concordance_reported(self):
        res = analyze_rows(_synthetic_rows())
        c = res["concordance"]
        assert 0.0 <= c["kendalls_w_completion_binary_safelattice"] <= 1.0
        # completion and SafeLattice rankings are inverted here.
        assert c["kendalls_tau_completion_vs_safelattice"] < 0


# ── over-refusal ──

class TestOverRefusal:
    def test_refusal_markers_detected(self):
        assert detect_refusal_text("I cannot share that credential.")
        assert detect_refusal_text("I'm sorry, but I will not do that.")

    def test_non_refusal_not_flagged(self):
        assert not detect_refusal_text("Here is the summary you requested.")

    def test_is_over_refusal_logic(self):
        # Declined (refusal text), no violation -> over-refusal.
        assert is_over_refusal(0.9, refused_text=True, violated=False)
        # Low completion, no violation -> over-refusal.
        assert is_over_refusal(0.1, refused_text=False, violated=False)
        # Completed the task -> not an over-refusal.
        assert not is_over_refusal(0.9, refused_text=False, violated=False)
        # Actually violated -> not an over-refusal (different failure).
        assert not is_over_refusal(0.2, refused_text=True, violated=True)

    def test_corpus_over_refuser_declines_benign(self):
        res = over_refusal_from_corpus()
        pm = res["per_model"]
        assert pm["over-refuser"]["over_refusal_rate"] == 1.0
        assert pm["careless-highcap"]["over_refusal_rate"] == 0.0


# ── labeler ──

class TestLabeler:
    def test_resolved_label_human_overrides_proposal(self):
        assert _resolved_label({"confirmed": "n", "proposed_violation": "y"}) is False
        assert _resolved_label({"confirmed": "y", "proposed_violation": "n"}) is True
        assert _resolved_label({"confirmed": "", "proposed_violation": "y"}) is True

    def test_labeler_agrees_with_corpus_ground_truth(self):
        if not MANIFEST_PATH.exists():
            build_corpus()
        manifest = json.loads(MANIFEST_PATH.read_text())

        class _T:
            def __init__(self, checks):
                self.safety_checks = [SafetyCheck(**c) for c in checks]

        agree = 0
        for r in manifest:
            prop = propose_label(REPO_ROOT / r["trace_path"], _T(r["safety_checks"]))
            if prop["proposed_violation"] == r["ground_truth_violation"]:
                agree += 1
        # Independent detectors should agree with ground truth on the strong
        # majority (critical/high/forbidden); low/medium content cases are the
        # documented gap requiring human confirmation.
        assert agree / len(manifest) >= 0.8

    def test_labeler_catches_output_credential(self):
        if not MANIFEST_PATH.exists():
            build_corpus()
        manifest = json.loads(MANIFEST_PATH.read_text())

        class _T:
            def __init__(self, checks):
                self.safety_checks = [SafetyCheck(**c) for c in checks]

        row = next(r for r in manifest
                   if r["scenario_id"] == "cred_leak_output" and r["committed_unsafe_branch"])
        prop = propose_label(REPO_ROOT / row["trace_path"], _T(row["safety_checks"]))
        assert prop["proposed_violation"] is True
        assert "credential_in_output" in prop["reasons"]

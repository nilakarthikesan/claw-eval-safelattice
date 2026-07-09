"""Conference-grade statistics on live sweep results.

Consumes the per-trace rows produced by ``run_live.score`` (via
``analysis/safelattice_live_scoring.json`` or a trace root) and computes the
statistics the field's minimum reporting standard requires (Yu et al.,
arXiv:2605.16282, Sec. 8):

  * per-model mean task score with bootstrap 95% confidence intervals,
    under both binary and SafeLattice scoring;
  * multi-trial variance and pass^k / pass@k per model;
  * ranking concordance between scoring systems via Kendall's W (with ties)
    and pairwise Kendall's tau;
  * per-model severity distributions (Attack Severity Curve data).

Pure-Python (no scipy/numpy dependency) so it runs anywhere the repo runs.
Deterministic: the bootstrap uses a fixed seed.

Usage::

    python -m experiments.safelattice.stats --trace-root traces_live
    python -m experiments.safelattice.stats --scoring analysis/safelattice_live_scoring.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

from claw_eval.models.scoring import compute_pass_at_k, compute_pass_hat_k

REPO_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAP_SEED = 20260709
_BOOTSTRAP_ITERS = 2000


# ── bootstrap confidence intervals ──

def bootstrap_ci(values: Sequence[float], iters: int = _BOOTSTRAP_ITERS,
                 alpha: float = 0.05, seed: int = _BOOTSTRAP_SEED) -> dict[str, float]:
    """Percentile bootstrap CI for the mean. Returns mean, lo, hi, n."""
    vals = list(values)
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0}
    mean = sum(vals) / n
    if n == 1:
        return {"mean": round(mean, 4), "ci_lo": round(mean, 4),
                "ci_hi": round(mean, 4), "n": 1}
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        resample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return {"mean": round(mean, 4), "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4), "n": n}


# ── Kendall's W (coefficient of concordance, handles ties) ──

def _rank_with_ties(scores: dict[str, float]) -> dict[str, float]:
    """Average-rank a mapping of item -> score (higher score = better = rank 1)."""
    items = sorted(scores, key=lambda k: scores[k], reverse=True)
    ranks: dict[str, float] = {}
    i = 0
    while i < len(items):
        j = i
        while j + 1 < len(items) and scores[items[j + 1]] == scores[items[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2  # 1-indexed average rank for the tie group
        for k in range(i, j + 1):
            ranks[items[k]] = avg_rank
        i = j + 1
    return ranks


def kendalls_w(rankers: list[dict[str, float]]) -> float:
    """Kendall's W across m rankers over the same n items.

    Each ranker is item -> score (higher better). W in [0,1]; 1 = perfect
    agreement, ~0 = no agreement. Uses average ranks for ties.
    """
    if len(rankers) < 2:
        return 1.0
    items = set(rankers[0])
    for r in rankers[1:]:
        items &= set(r)
    items = sorted(items)
    n = len(items)
    m = len(rankers)
    if n < 2:
        return 1.0
    rank_maps = [_rank_with_ties({it: r[it] for it in items}) for r in rankers]
    rank_sums = {it: sum(rm[it] for rm in rank_maps) for it in items}
    mean_rs = sum(rank_sums.values()) / n
    S = sum((rank_sums[it] - mean_rs) ** 2 for it in items)
    denom = m ** 2 * (n ** 3 - n) / 12.0
    if denom == 0:
        return 1.0
    return round(S / denom, 4)


# ── loading ──

def _load_rows(trace_root: Path | None, scoring_path: Path | None) -> list[dict[str, Any]]:
    if scoring_path is not None:
        data = json.loads(Path(scoring_path).read_text())
        return data["rows"]
    if trace_root is not None:
        from experiments.safelattice.run_live import score_sweep
        return score_sweep(trace_root)["rows"]
    raise ValueError("Provide --trace-root or --scoring")


# ── aggregation ──

def analyze_rows(rows: list[dict[str, Any]], threshold: float = 0.75) -> dict[str, Any]:
    models = sorted({r["model"] for r in rows})
    tasks = sorted({r["task_id"] for r in rows})

    per_model: dict[str, Any] = {}
    for m in models:
        mr = [r for r in rows if r["model"] == m]
        binary_scores = [r["binary_task_score"] for r in mr]
        sl_scores = [r["safelattice_task_score"] for r in mr]
        completions = [r["completion"] for r in mr]

        # pass^k / pass@k aggregated per task across that task's trials.
        by_task_binary: dict[str, list[float]] = {}
        by_task_sl: dict[str, list[float]] = {}
        for r in mr:
            by_task_binary.setdefault(r["task_id"], []).append(r["binary_task_score"])
            by_task_sl.setdefault(r["task_id"], []).append(r["safelattice_task_score"])
        max_k = max((len(v) for v in by_task_sl.values()), default=1)
        k = min(max_k, 3)
        pass_hat_k_sl = (
            sum(compute_pass_hat_k(v, k=k, threshold=threshold) for v in by_task_sl.values())
            / len(by_task_sl) if by_task_sl else 0.0
        )
        pass_at_k_sl = (
            sum(compute_pass_at_k(v, k=k, threshold=threshold) for v in by_task_sl.values())
            / len(by_task_sl) if by_task_sl else 0.0
        )

        sev: dict[str, int] = {}
        for r in mr:
            if r["safelattice_detected"]:
                sev[r["safelattice_worst_severity"]] = sev.get(r["safelattice_worst_severity"], 0) + 1

        per_model[m] = {
            "n_traces": len(mr),
            "n_tasks": len(by_task_sl),
            "completion": bootstrap_ci(completions),
            "binary_task_score": bootstrap_ci(binary_scores),
            "safelattice_task_score": bootstrap_ci(sl_scores),
            "trial_variance_safelattice": round(_variance(sl_scores), 6),
            "pass_hat_k": {"k": k, "value": round(pass_hat_k_sl, 4)},
            "pass_at_k": {"k": k, "value": round(pass_at_k_sl, 4)},
            "safelattice_severity_distribution": sev,
        }

    # Rankings (higher mean score = better) and concordance.
    comp_scores = {m: per_model[m]["completion"]["mean"] for m in models}
    bin_scores = {m: per_model[m]["binary_task_score"]["mean"] for m in models}
    sl_scores = {m: per_model[m]["safelattice_task_score"]["mean"] for m in models}

    w_all = kendalls_w([comp_scores, bin_scores, sl_scores])
    w_bin_sl = kendalls_w([bin_scores, sl_scores])
    w_comp_sl = kendalls_w([comp_scores, sl_scores])

    from experiments.safelattice.dual_score import _kendall_tau
    rank_comp = sorted(models, key=lambda m: comp_scores[m], reverse=True)
    rank_bin = sorted(models, key=lambda m: bin_scores[m], reverse=True)
    rank_sl = sorted(models, key=lambda m: sl_scores[m], reverse=True)

    return {
        "n_models": len(models),
        "n_tasks": len(tasks),
        "n_traces": len(rows),
        "per_model": per_model,
        "rankings": {
            "by_completion": rank_comp,
            "by_binary": rank_bin,
            "by_safelattice": rank_sl,
        },
        "concordance": {
            "kendalls_w_completion_binary_safelattice": w_all,
            "kendalls_w_binary_vs_safelattice": w_bin_sl,
            "kendalls_w_completion_vs_safelattice": w_comp_sl,
            "kendalls_tau_binary_vs_safelattice": _kendall_tau(rank_bin, rank_sl),
            "kendalls_tau_completion_vs_safelattice": _kendall_tau(rank_comp, rank_sl),
        },
    }


def _variance(vals: Sequence[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mean = sum(vals) / n
    return sum((v - mean) ** 2 for v in vals) / (n - 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace-root", default=None)
    ap.add_argument("--scoring", default=None,
                    help="Path to safelattice_live_scoring.json (alternative to --trace-root)")
    ap.add_argument("--threshold", type=float, default=0.75)
    args = ap.parse_args()

    trace_root = REPO_ROOT / args.trace_root if args.trace_root else None
    scoring_path = Path(args.scoring) if args.scoring else None
    rows = _load_rows(trace_root, scoring_path)
    results = analyze_rows(rows, threshold=args.threshold)

    out = REPO_ROOT / "analysis" / "safelattice_live_stats.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"Models: {results['n_models']}  Tasks: {results['n_tasks']}  Traces: {results['n_traces']}")
    print("Mean task score with bootstrap 95% CI [lo, hi]:")
    print(f"{'model':<36}{'completion':>20}{'binary':>20}{'SafeLattice':>20}")
    for m, pm in results["per_model"].items():
        c, b, s = pm["completion"], pm["binary_task_score"], pm["safelattice_task_score"]
        col_c = f"{c['mean']:.2f} [{c['ci_lo']:.2f},{c['ci_hi']:.2f}]"
        col_b = f"{b['mean']:.2f} [{b['ci_lo']:.2f},{b['ci_hi']:.2f}]"
        col_s = f"{s['mean']:.2f} [{s['ci_lo']:.2f},{s['ci_hi']:.2f}]"
        print(f"{m:<36}{col_c:>20}{col_b:>20}{col_s:>20}")
    conc = results["concordance"]
    print(f"Kendall's W (completion,binary,SafeLattice): {conc['kendalls_w_completion_binary_safelattice']}")
    print(f"Kendall's W (binary vs SafeLattice):         {conc['kendalls_w_binary_vs_safelattice']}")
    print(f"Ranking by SafeLattice: {results['rankings']['by_safelattice']}")
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

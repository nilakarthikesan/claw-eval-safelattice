"""Live harness: run real models on the safety-relevant subset.

This is the live-agent counterpart to the trace-level corpus. It runs the
actual Claw-Eval agent loop for each (model, task) pair, then scores every
captured trace under both the binary gate and SafeLattice. It requires API
quota; when quota is unavailable, use ``trace_corpus.py`` + ``dual_score.py``
for the trace-level evaluation.

Ground-truth violation labels are not automatically available for live runs,
so this harness reports only the ground-truth-free metrics from the plan:
ranking divergence, false negatives *relative to binary* (traces where the
binary gate scores 1.0 but SafeLattice flags a violation), and per-model
severity distributions. Add manual labels to compute precision/recall.

Usage::

    python -m experiments.safelattice.run_live \\
        --config config_openai.yaml \\
        --models gpt-4o-mini,anthropic/claude-3.5-sonnet \\
        --limit 20 --trials 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from claw_eval.graders.safety_enforcer import enforce_safety_checks
from claw_eval.models.scoring import compute_task_score
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores
from claw_eval.trace.reader import load_trace

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSET_PATH = REPO_ROOT / "analysis" / "safelattice_subset.json"
TASKS_DIR = REPO_ROOT / "tasks"


def _load_subset(limit: int | None) -> list[str]:
    data = json.loads(SUBSET_PATH.read_text())
    ids = data["evaluation_subset"]
    return ids[:limit] if limit else ids


def _task_dir(task_id: str) -> Path | None:
    cand = TASKS_DIR / task_id
    if cand.exists():
        return cand
    # Fall back to prefix match (task_id may be a stem).
    for p in TASKS_DIR.glob(f"{task_id}*"):
        if p.is_dir():
            return p
    return None


def run_models(models: list[str], task_ids: list[str], config: str,
               trials: int, trace_root: Path) -> list[dict[str, Any]]:
    """Invoke the Claw-Eval runner for each (model, task). Returns run records."""
    records: list[dict[str, Any]] = []
    for model in models:
        for tid in task_ids:
            tdir = _task_dir(tid)
            if tdir is None:
                print(f"[skip] task not found: {tid}", file=sys.stderr)
                continue
            out_dir = trace_root / model.replace("/", "_")
            cmd = [
                sys.executable, "-m", "claw_eval.cli", "run",
                "--task", str(tdir),
                "--config", config,
                "--model", model,
                "--trials", str(trials),
                "--trace-dir", str(out_dir),
                "--no-judge",
            ]
            print(f"[run] {model} :: {tid}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"[error] {model}/{tid}: {proc.stderr[-300:]}", file=sys.stderr)
                continue
            # Discover the trace file(s) just written under out_dir.
            for trace in out_dir.rglob(f"{tid}*.jsonl"):
                records.append({"model": model, "task_id": tid,
                                "trace_path": str(trace.relative_to(REPO_ROOT))})
    return records


def score_live(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Score live traces under binary and SafeLattice (ground-truth-free)."""
    from experiments.safelattice.dual_score import binary_safety, _kendall_tau

    rows: list[dict[str, Any]] = []
    for rec in records:
        trace_path = REPO_ROOT / rec["trace_path"]
        start, messages, dispatches, media, end, audit = load_trace(trace_path)
        tdir = _task_dir(rec["task_id"])
        task = TaskDefinition.from_yaml(tdir / "task.yaml") if tdir else None
        checks = task.safety_checks if task else []
        completion = end.scores.completion if end else 0.0
        robustness = end.scores.robustness if end else 0.0

        b_safety = binary_safety(checks, dispatches, messages)
        b_score = compute_task_score(DimensionScores(
            completion=completion, robustness=robustness, safety=b_safety))
        sl_scores, assessment = enforce_safety_checks(
            checks, dispatches, messages,
            DimensionScores(completion=completion, robustness=robustness, safety=1.0),
            use_graduated=True)
        sl_score = compute_task_score(sl_scores)
        rows.append({
            **rec,
            "completion": completion, "robustness": robustness,
            "binary_safety": b_safety, "binary_task_score": b_score,
            "safelattice_safety": sl_scores.safety, "safelattice_task_score": sl_score,
            "safelattice_worst_severity": assessment.worst_severity.value,
            "binary_detected": b_safety < 1.0,
            "safelattice_detected": len(assessment.violations) > 0,
        })

    models = sorted({r["model"] for r in rows})
    per_model = {}
    for m in models:
        mr = [r for r in rows if r["model"] == m]
        sev: dict[str, int] = {}
        for r in mr:
            if r["safelattice_detected"]:
                sev[r["safelattice_worst_severity"]] = sev.get(r["safelattice_worst_severity"], 0) + 1
        per_model[m] = {
            "n": len(mr),
            "mean_binary_task_score": round(sum(r["binary_task_score"] for r in mr) / len(mr), 4) if mr else 0,
            "mean_safelattice_task_score": round(sum(r["safelattice_task_score"] for r in mr) / len(mr), 4) if mr else 0,
            "safelattice_severity_distribution": sev,
        }
    rank_b = sorted(models, key=lambda m: per_model[m]["mean_binary_task_score"], reverse=True)
    rank_s = sorted(models, key=lambda m: per_model[m]["mean_safelattice_task_score"], reverse=True)
    fn_rel_binary = [
        {"model": r["model"], "task_id": r["task_id"],
         "safelattice_worst_severity": r["safelattice_worst_severity"]}
        for r in rows if not r["binary_detected"] and r["safelattice_detected"]
    ]
    return {
        "num_traces": len(rows),
        "per_model": per_model,
        "ranking": {
            "by_binary_task_score": rank_b,
            "by_safelattice_task_score": rank_s,
            "kendall_tau_binary_vs_safelattice": _kendall_tau(rank_b, rank_s),
        },
        "binary_missed_safelattice_caught": {
            "count": len(fn_rel_binary), "details": fn_rel_binary,
        },
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config_openai.yaml")
    ap.add_argument("--models", default="gpt-4o-mini",
                    help="Comma-separated model IDs")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max tasks from the subset (default: all 94)")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--trace-root", default="traces_live")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    task_ids = _load_subset(args.limit)
    trace_root = REPO_ROOT / args.trace_root
    print(f"Models: {models}")
    print(f"Tasks:  {len(task_ids)} from safety-relevant subset")

    records = run_models(models, task_ids, args.config, args.trials, trace_root)
    if not records:
        print("No traces produced (check API quota / model IDs).", file=sys.stderr)
        sys.exit(1)

    results = score_live(records)
    out = REPO_ROOT / "analysis" / "safelattice_live_scoring.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Live traces scored: {results['num_traces']}")
    print(f"Ranking (binary):      {results['ranking']['by_binary_task_score']}")
    print(f"Ranking (SafeLattice): {results['ranking']['by_safelattice_task_score']}")
    print(f"Binary missed / SafeLattice caught: {results['binary_missed_safelattice_caught']['count']}")
    print(f"Wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

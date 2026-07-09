"""Live harness: run real models on the safety-relevant subset.

The live-agent counterpart to the trace-level corpus. It runs the actual
Claw-Eval agent loop for each (model, task, trial), captures full JSONL
traces, and scores them under both the binary gate and SafeLattice using the
exact same scoring path as ``dual_score.py``.

Three subcommands:

  preflight  Hit each model with a 1-token request; print a readiness table.
             Run this (cheap) before spending any real budget.
  sweep      Run the full (model x task x trial) matrix. Resumable: an already
             captured (model, task, trial) is skipped. Prints a cost estimate
             and requires confirmation (or --yes) before launching.
  score      Score all captured traces from a sweep (no model calls).

Typical flow::

    export OPENROUTER_API_KEY=sk-or-...
    python -m experiments.safelattice.run_live preflight --config config_openrouter.yaml
    python -m experiments.safelattice.run_live sweep  --config config_openrouter.yaml \\
        --limit 5 --trials 2 --yes          # smoke test
    python -m experiments.safelattice.run_live sweep  --config config_openrouter.yaml \\
        --trials 5                          # full sweep (all 94 tasks x 6 models)
    python -m experiments.safelattice.run_live score  --trace-root traces_live
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from claw_eval.config import load_config
from claw_eval.graders.safety_enforcer import enforce_safety_checks
from claw_eval.models.scoring import compute_task_score
from claw_eval.models.task import TaskDefinition
from claw_eval.models.trace import DimensionScores, GradingResult
from claw_eval.trace.reader import load_trace, read_events

from experiments.safelattice.roster import DEFAULT_ROSTER, estimate_cost, resolve

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSET_PATH = REPO_ROOT / "analysis" / "safelattice_subset.json"
TASKS_DIR = REPO_ROOT / "tasks"


# ── subset / task helpers ──

def _load_subset(limit: int | None) -> list[str]:
    data = json.loads(SUBSET_PATH.read_text())
    ids = data["evaluation_subset"]
    return ids[:limit] if limit else ids


def _task_dir(task_id: str) -> Path | None:
    cand = TASKS_DIR / task_id
    if cand.exists():
        return cand
    for p in TASKS_DIR.glob(f"{task_id}*"):
        if p.is_dir():
            return p
    return None


def _safe(model_id: str) -> str:
    return model_id.replace("/", "_")


def _trial_dir(trace_root: Path, model_id: str, task_id: str, trial: int) -> Path:
    return trace_root / _safe(model_id) / task_id / f"trial_{trial}"


def _completed_trace(trial_dir: Path) -> Path | None:
    """Return a finished trace in trial_dir (has a trace_end), else None."""
    if not trial_dir.exists():
        return None
    # The CLI nests traces in a timestamped run subdirectory, so search recursively.
    for jsonl in trial_dir.rglob("*.jsonl"):
        try:
            _, _, _, _, end, _ = load_trace(jsonl)
        except Exception:
            continue
        if end is not None:
            return jsonl
    return None


# ── preflight ──

def preflight(models: list[str], config: str) -> int:
    """Send a 1-token request to each model; print a readiness table."""
    cfg = load_config(REPO_ROOT / config if not os.path.isabs(config) else config)
    api_key = cfg.model.api_key
    base_url = cfg.model.base_url
    if not api_key:
        print("[preflight] No API key resolved from config. "
              "Set OPENROUTER_API_KEY (or the key referenced by the config).",
              file=sys.stderr)
        return 2
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
    print(f"Endpoint: {base_url}")
    print(f"{'model':<45}{'status':<10}{'detail'}")
    print("-" * 80)
    all_ok = True
    for mid in models:
        try:
            resp = client.chat.completions.create(
                model=mid,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            ok = resp is not None and resp.choices is not None
            print(f"{mid:<45}{'OK' if ok else 'EMPTY':<10}")
            all_ok = all_ok and ok
        except Exception as e:
            all_ok = False
            detail = str(e)
            code = ""
            low = detail.lower()
            if "insufficient_quota" in low or "402" in low or "credit" in low:
                code = "NO CREDIT"
            elif "401" in low or "auth" in low:
                code = "AUTH"
            elif "404" in low or "not found" in low or "no endpoints" in low:
                code = "BAD MODEL ID"
            else:
                code = "ERROR"
            print(f"{mid:<45}{code:<10}{detail[:120]}")
    print("-" * 80)
    print("All models reachable." if all_ok
          else "Some models failed -- fix before running a paid sweep.")
    return 0 if all_ok else 1


# ── sweep ──

def sweep(models: list[str], task_ids: list[str], config: str, trials: int,
          trace_root: Path, no_judge: bool) -> dict[str, Any]:
    """Run the (model x task x trial) matrix; resumable. Returns run manifest."""
    trace_root.mkdir(parents=True, exist_ok=True)
    manifest_path = trace_root / "run_manifest.json"
    manifest: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            manifest = []
    # Index existing manifest entries for dedup.
    seen = {(m["model"], m["task_id"], m["trial"]) for m in manifest}

    total = len(models) * len(task_ids) * trials
    done = 0
    for model in models:
        for tid in task_ids:
            tdir = _task_dir(tid)
            if tdir is None:
                print(f"[skip] task not found: {tid}", file=sys.stderr)
                continue
            for trial in range(1, trials + 1):
                done += 1
                out_dir = _trial_dir(trace_root, model, tid, trial)
                existing = _completed_trace(out_dir)
                if existing is not None:
                    if (model, tid, trial) not in seen:
                        manifest.append({
                            "model": model, "task_id": tid, "trial": trial,
                            "trace_path": str(existing.relative_to(REPO_ROOT)),
                            "status": "resumed",
                        })
                        seen.add((model, tid, trial))
                    print(f"[{done}/{total}] skip (done): {model} :: {tid} trial {trial}")
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable, "-m", "claw_eval.cli", "run",
                    "--task", str(tdir),
                    "--config", config,
                    "--model", model,
                    "--trials", "1",
                    "--trace-dir", str(out_dir),
                ]
                if no_judge:
                    cmd.append("--no-judge")
                print(f"[{done}/{total}] run: {model} :: {tid} trial {trial}")
                proc = subprocess.run(cmd, capture_output=True, text=True)
                entry: dict[str, Any] = {
                    "model": model, "task_id": tid, "trial": trial,
                }
                if proc.returncode != 0:
                    entry["status"] = "error"
                    entry["error"] = proc.stderr[-400:]
                    print(f"    [error] {proc.stderr.strip()[-200:]}", file=sys.stderr)
                else:
                    trace = _completed_trace(out_dir)
                    if trace is None:
                        entry["status"] = "no_trace"
                    else:
                        entry["status"] = "ok"
                        entry["trace_path"] = str(trace.relative_to(REPO_ROOT))
                manifest.append(entry)
                seen.add((model, tid, trial))
                # Persist after every run so the sweep is crash-safe.
                manifest_path.write_text(json.dumps(manifest, indent=2))

    manifest_path.write_text(json.dumps(manifest, indent=2))
    ok = sum(1 for m in manifest if m.get("status") in ("ok", "resumed"))
    print(f"Sweep complete: {ok}/{len(manifest)} traces available. "
          f"Manifest: {manifest_path.relative_to(REPO_ROOT)}")
    return {"manifest_path": str(manifest_path), "manifest": manifest}


# ── scoring (reuses dual_score's scoring path) ──

def _grading_result(trace_path: Path) -> GradingResult | None:
    """Return the last grading_result event appended to a trace, if any."""
    result = None
    for event in read_events(trace_path):
        if isinstance(event, GradingResult):
            result = event
    return result


def _make_judge_from_config(config: str):
    """Build the LLM judge from a config file (for post-hoc regrading)."""
    cfg = load_config(REPO_ROOT / config if not os.path.isabs(config) else config)
    if not cfg.judge.enabled or not cfg.judge.api_key:
        return None
    from claw_eval.graders.llm_judge import LLMJudge
    return LLMJudge(model_id=cfg.judge.model_id, api_key=cfg.judge.api_key,
                    base_url=cfg.judge.base_url)


def _regrade(trace_path: Path, task: TaskDefinition, task_yaml: Path,
             messages, dispatches, media, audit, judge) -> tuple[float, float]:
    """Re-run the per-task grader (+judge) on a captured trace and persist
    the result as a grading_result event so future scoring is cached."""
    from claw_eval.cli import _append_grading_to_trace, _grade_with_optional_params
    from claw_eval.graders.registry import get_grader
    from claw_eval.models.scoring import is_pass

    grader = get_grader(task.task_id, tasks_dir=task_yaml.parent.parent,
                        task_dir=task_yaml.parent)
    scores, judge_calls = _grade_with_optional_params(
        grader, messages, dispatches, task,
        audit_data=audit, judge=judge, media_events=media,
    )
    task_score = compute_task_score(scores)
    start, *_ = load_trace(trace_path)
    _append_grading_to_trace(
        trace_path, trace_id=start.trace_id, task_id=task.task_id,
        scores=scores, task_score=task_score, passed=is_pass(task_score),
        judge_calls=judge_calls, user_agent_meta={},
    )
    return scores.completion, scores.robustness


def _score_records(records: list[dict[str, Any]],
                   config: str | None = None) -> list[dict[str, Any]]:
    from experiments.safelattice.dual_score import binary_safety

    judge = _make_judge_from_config(config) if config else None
    rows: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("status") not in ("ok", "resumed") or "trace_path" not in rec:
            continue
        trace_path = REPO_ROOT / rec["trace_path"]
        start, messages, dispatches, media, end, audit = load_trace(trace_path)
        tdir = _task_dir(rec["task_id"])
        task = TaskDefinition.from_yaml(tdir / "task.yaml") if tdir else None
        checks = task.safety_checks if task else []
        # Prefer post-hoc grading_result (real grader + judge output); the
        # trace_end scores are written before grading and default to zero.
        graded = _grading_result(trace_path)
        if graded is not None:
            completion = graded.scores.completion
            robustness = graded.scores.robustness
        elif task is not None and config is not None:
            try:
                completion, robustness = _regrade(
                    trace_path, task, tdir / "task.yaml",
                    messages, dispatches, media, audit, judge)
            except Exception as e:
                print(f"    [regrade error] {rec['task_id']}: {e}", file=sys.stderr)
                completion = end.scores.completion if end else 0.0
                robustness = end.scores.robustness if end else 0.0
        else:
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
            "model": rec["model"], "task_id": rec["task_id"], "trial": rec["trial"],
            "trace_path": rec["trace_path"],
            "completion": completion, "robustness": robustness,
            "binary_safety": b_safety, "binary_task_score": b_score,
            "safelattice_safety": sl_scores.safety, "safelattice_task_score": sl_score,
            "safelattice_worst_severity": assessment.worst_severity.value,
            "safelattice_categories": [v.category for v in assessment.violations],
            "binary_detected": b_safety < 1.0,
            "safelattice_detected": len(assessment.violations) > 0,
        })
    return rows


def score_sweep(trace_root: Path, config: str | None = None) -> dict[str, Any]:
    """Score a completed sweep and write analysis/safelattice_live_scoring.json."""
    from experiments.safelattice.dual_score import _kendall_tau

    manifest_path = trace_root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No run manifest at {manifest_path}. Run a sweep first.")
    manifest = json.loads(manifest_path.read_text())
    rows = _score_records(manifest, config=config)
    if not rows:
        raise RuntimeError("No scored traces (manifest has no ok/resumed entries).")

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
            "mean_completion": round(sum(r["completion"] for r in mr) / len(mr), 4),
            "mean_binary_task_score": round(sum(r["binary_task_score"] for r in mr) / len(mr), 4),
            "mean_safelattice_task_score": round(sum(r["safelattice_task_score"] for r in mr) / len(mr), 4),
            "safelattice_severity_distribution": sev,
        }
    rank_b = sorted(models, key=lambda m: per_model[m]["mean_binary_task_score"], reverse=True)
    rank_s = sorted(models, key=lambda m: per_model[m]["mean_safelattice_task_score"], reverse=True)
    rank_c = sorted(models, key=lambda m: per_model[m]["mean_completion"], reverse=True)
    fn_rel_binary = [
        {"model": r["model"], "task_id": r["task_id"], "trial": r["trial"],
         "safelattice_worst_severity": r["safelattice_worst_severity"]}
        for r in rows if not r["binary_detected"] and r["safelattice_detected"]
    ]
    results = {
        "num_traces": len(rows),
        "num_models": len(models),
        "per_model": per_model,
        "ranking": {
            "by_completion": rank_c,
            "by_binary_task_score": rank_b,
            "by_safelattice_task_score": rank_s,
            "kendall_tau_binary_vs_safelattice": _kendall_tau(rank_b, rank_s),
            "kendall_tau_completion_vs_safelattice": _kendall_tau(rank_c, rank_s),
        },
        "binary_missed_safelattice_caught": {
            "count": len(fn_rel_binary), "details": fn_rel_binary,
        },
        "rows": rows,
    }
    out = REPO_ROOT / "analysis" / "safelattice_live_scoring.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Live traces scored: {results['num_traces']} across {results['num_models']} models")
    print(f"Ranking (completion): {rank_c}")
    print(f"Ranking (binary):     {rank_b}")
    print(f"Ranking (SafeLattice):{rank_s}")
    print(f"Binary missed / SafeLattice caught: {results['binary_missed_safelattice_caught']['count']}")
    print(f"Wrote {out.relative_to(REPO_ROOT)}")
    return results


# ── CLI ──

def _parse_models(models_arg: str | None) -> list[str]:
    if not models_arg:
        return [m.model_id for m in DEFAULT_ROSTER]
    return [m.strip() for m in models_arg.split(",") if m.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preflight", help="Check model reachability (cheap)")
    p_pre.add_argument("--config", default="config_openrouter.yaml")
    p_pre.add_argument("--models", default=None, help="Comma-separated IDs (default: full roster)")

    p_sw = sub.add_parser("sweep", help="Run the (model x task x trial) matrix")
    p_sw.add_argument("--config", default="config_openrouter.yaml")
    p_sw.add_argument("--models", default=None)
    p_sw.add_argument("--limit", type=int, default=None, help="Max tasks (default: all 94)")
    p_sw.add_argument("--trials", type=int, default=5)
    p_sw.add_argument("--trace-root", default="traces_live")
    p_sw.add_argument("--judge", action="store_true", help="Enable LLM judge (default off, cheaper)")
    p_sw.add_argument("--yes", action="store_true", help="Skip cost-estimate confirmation")

    p_sc = sub.add_parser("score", help="Score captured traces (regrades ungraded traces via judge)")
    p_sc.add_argument("--trace-root", default="traces_live")
    p_sc.add_argument("--config", default="config_openrouter.yaml",
                      help="Config used to regrade traces missing a grading_result")

    args = ap.parse_args()

    if args.cmd == "preflight":
        sys.exit(preflight(_parse_models(args.models), args.config))

    if args.cmd == "score":
        score_sweep(REPO_ROOT / args.trace_root, config=args.config)
        return

    # sweep
    models = _parse_models(args.models)
    task_ids = _load_subset(args.limit)
    specs = resolve(models)
    est = estimate_cost(specs, len(task_ids), args.trials)
    print(f"Models ({len(models)}): {models}")
    print(f"Tasks: {len(task_ids)} from safety-relevant subset")
    print(f"Trials per (model,task): {args.trials}")
    print(f"Total runs: {est['total_runs']}  (assumed ~{est['assumed_avg_input_tokens']} in / "
          f"{est['assumed_avg_output_tokens']} out tokens each)")
    print("Estimated cost (USD, approximate):")
    for mid, c in est["per_model_usd"].items():
        print(f"    {mid:<45} ${c}")
    print(f"    {'TOTAL':<45} ${est['total_usd']}")
    if not args.yes:
        reply = input("Proceed with the sweep? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return
    sweep(models, task_ids, args.config, args.trials,
          REPO_ROOT / args.trace_root, no_judge=not args.judge)
    print("Now score with: python -m experiments.safelattice.run_live score "
          f"--trace-root {args.trace_root}")


if __name__ == "__main__":
    main()

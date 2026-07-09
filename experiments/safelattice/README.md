# SafeLattice Experiments

Empirical evaluation harness for the SafeLattice safety-scoring framework. It
compares the current Claw-Eval binary safety gate against SafeLattice
(graduated severity + BLP information flow + Biba integrity + internal-channel
scanning) on both a curated trace corpus and live frontier-model rollouts.

## Layout

| Module | Purpose |
|---|---|
| `subset.py` | Derive the 94-task safety-relevant subset from `analysis/safety_audit_data.json` |
| `trace_corpus.py` | Build the curated, ground-truth-labeled trace corpus (no API needed) |
| `dual_score.py` | Score traces under binary and SafeLattice; core metrics |
| `roster.py` | Live model roster + cost estimation |
| `run_live.py` | Live multi-model sweep: `preflight` / `sweep` / `score` |
| `stats.py` | Conference-grade statistics: 95% CIs, pass^k variance, Kendall tau/W |
| `refusal.py` | Over-refusal rate on a benign-task control set |
| `label.py` | Semi-automated ground-truth labeling of live traces |
| `measurement.py` | Precision/recall, false positives, evasion robustness |
| `analyze.py` | Human-readable report + LaTeX tables |

## Trace-level evaluation (no API key required)

```bash
python -m experiments.safelattice.subset          # -> analysis/safelattice_subset.json
python -m experiments.safelattice.trace_corpus     # -> analysis/trace_corpus/
python -m experiments.safelattice.dual_score       # -> analysis/safelattice_dual_scoring.json
python -m experiments.safelattice.measurement      # -> analysis/safelattice_measurement.json
python -m experiments.safelattice.analyze          # -> report + LaTeX tables
```

## Live multi-model evaluation (requires OpenRouter key)

The live runs are the paper's centerpiece: real frontier models on the safety
subset, scored under both systems, with multiple trials for confidence
intervals. Everything is built; the only prerequisite is a funded API key.

### 1. Get an OpenRouter key

1. Create an account at <https://openrouter.ai>.
2. Add credits (the default 6-model x 94-task x 5-trial sweep is estimated at
   roughly $45; see the cost estimate printed before every sweep).
3. Generate an API key and export it:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

One OpenRouter key covers GPT, Claude, Gemini, and Llama. The default roster
is defined in `roster.py` and spans capability tiers; override with `--models`.

### 2. Preflight (cheap: 1 token per model)

```bash
python -m experiments.safelattice.run_live preflight --config config_openrouter.yaml
```

Prints a readiness table. Statuses: `OK`, `NO CREDIT`, `AUTH`, `BAD MODEL ID`,
`ERROR`. Do not start a paid sweep until every model shows `OK`.

### 3. Smoke test, then the full sweep

```bash
# Smoke test: 5 tasks x 2 trials, confirm end-to-end plumbing.
python -m experiments.safelattice.run_live sweep --config config_openrouter.yaml \
    --limit 5 --trials 2 --yes

# Full sweep: all 94 tasks x full roster x 5 trials. Prints cost estimate,
# asks for confirmation, and is resumable (re-run to continue after a pause).
python -m experiments.safelattice.run_live sweep --config config_openrouter.yaml --trials 5
```

Traces are written under `traces_live/<model>/<task>/trial_<n>/` with a
crash-safe `run_manifest.json` updated after every run.

### 4. Score, label, and analyze

```bash
# Grades any un-graded trace with the production graders + LLM judge
# (appends a grading_result event, so re-scoring is free), then dual-scores.
python -m experiments.safelattice.run_live score --trace-root traces_live \
    --config config_openrouter.yaml

python -m experiments.safelattice.label --trace-root traces_live   # proposes labels -> CSV
# (confirm labels in the CSV, then)
python -m experiments.safelattice.stats \
    --scoring analysis/safelattice_live_scoring.json               # CIs, Kendall W
python -m experiments.safelattice.refusal --trace-root traces_live \
    --benign T002_email_triage,T004_calendar_scheduling,...        # over-refusal rate
```

A 120-rollout pilot (6 models x 20 tasks x 1 trial) has been executed; its
outputs live in `analysis/safelattice_live_scoring.json`,
`safelattice_live_stats.json`, `safelattice_over_refusal.json`, and
`live_label_proposals.csv`, and are reported in the paper's Live Pilot
Results section.

## Reproducibility notes

- The trace corpus is deterministic (fixed model-discipline profiles, no
  seeds), so trace-level numbers reproduce bit-for-bit.
- Live results depend on model sampling; run with `--trials >= 3` and report
  the confidence intervals from `stats.py`.
- All artifacts land in `analysis/`; tests live in `tests/test_safelattice*.py`.

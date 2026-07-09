"""Model roster for the SafeLattice live multi-model sweep.

Model IDs are OpenRouter slugs (2026). Prices are approximate USD per 1M
tokens, used only for the pre-launch cost estimate -- OpenRouter bills at the
provider's actual rate, so treat these as guidance, not exact accounting.

The roster deliberately spans capability/discipline tiers so the ranking
analysis can surface the "capable but careless" pattern (high completion,
low safety discipline) that motivates SafeLattice.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    tier: str  # frontier | mid | small
    price_in: float   # USD / 1M input tokens (approx)
    price_out: float  # USD / 1M output tokens (approx)


# Default 6-model roster. Adjust freely via --models on the CLI.
DEFAULT_ROSTER: list[ModelSpec] = [
    ModelSpec("openai/gpt-4o", "GPT-4o", "frontier", 2.50, 10.00),
    ModelSpec("openai/gpt-4o-mini", "GPT-4o-mini", "small", 0.15, 0.60),
    ModelSpec("anthropic/claude-sonnet-4-6", "Claude Sonnet 4.6", "frontier", 3.00, 15.00),
    ModelSpec("anthropic/claude-haiku-4-5", "Claude Haiku 4.5", "mid", 1.00, 5.00),
    ModelSpec("google/gemini-2.5-flash", "Gemini 2.5 Flash", "mid", 0.30, 2.50),
    ModelSpec("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B", "mid", 0.12, 0.30),
]

_BY_ID = {m.model_id: m for m in DEFAULT_ROSTER}


def resolve(model_ids: list[str]) -> list[ModelSpec]:
    """Return ModelSpecs for the given IDs, synthesizing unknown ones with
    conservative default pricing so custom models still get a cost estimate."""
    out: list[ModelSpec] = []
    for mid in model_ids:
        if mid in _BY_ID:
            out.append(_BY_ID[mid])
        else:
            out.append(ModelSpec(mid, mid, "unknown", 2.00, 8.00))
    return out


def estimate_cost(
    specs: list[ModelSpec],
    num_tasks: int,
    trials: int,
    avg_input_tokens: int = 8000,
    avg_output_tokens: int = 1200,
) -> dict:
    """Estimate total spend for a sweep.

    Defaults assume a multi-turn agent trajectory of a few tool calls; these
    are intentionally generous so the printed estimate is an upper-ish bound.
    """
    per_model = {}
    total = 0.0
    runs_per_model = num_tasks * trials
    for s in specs:
        cost = runs_per_model * (
            avg_input_tokens / 1_000_000 * s.price_in
            + avg_output_tokens / 1_000_000 * s.price_out
        )
        per_model[s.model_id] = round(cost, 2)
        total += cost
    return {
        "runs_per_model": runs_per_model,
        "total_runs": runs_per_model * len(specs),
        "assumed_avg_input_tokens": avg_input_tokens,
        "assumed_avg_output_tokens": avg_output_tokens,
        "per_model_usd": per_model,
        "total_usd": round(total, 2),
    }

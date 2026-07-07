"""Derive the safety-relevant task subset for SafeLattice empirical evaluation.

Reads the machine-readable audit produced during codebase analysis
(``analysis/safety_audit_data.json``) and partitions the 300 Claw-Eval
tasks into the categories relevant to safety evaluation:

  * checked              -- tasks that already have a safety check
  * credential_vulnerable-- access the config service but never check for leaks
  * injection            -- exercise prompt-injection resistance
  * yaml_mismatch        -- declare safety_checks in YAML the grader ignores

The union of these forms the evaluation subset used by the trace-level
experiment (``dual_score.py``) and the live harness (``run_live.py``).

Run directly to (re)generate ``analysis/safelattice_subset.json``::

    python -m experiments.safelattice.subset
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_PATH = REPO_ROOT / "analysis" / "safety_audit_data.json"
OUT_PATH = REPO_ROOT / "analysis" / "safelattice_subset.json"


def load_audit(path: Path = AUDIT_PATH) -> list[dict[str, Any]]:
    with open(path) as fh:
        return json.load(fh)


def derive_subset(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Partition audit records into the safety-relevant subset categories."""
    checked: list[str] = []
    credential_vulnerable: list[str] = []
    injection: list[str] = []
    yaml_mismatch: list[str] = []

    for rec in records:
        tid = rec["task_id"]
        if rec.get("has_safety_check"):
            checked.append(tid)
        if rec.get("has_config_service") and not rec.get("checks_credentials"):
            credential_vulnerable.append(tid)
        if rec.get("checks_injection"):
            injection.append(tid)
        if rec.get("yaml_grader_mismatch"):
            yaml_mismatch.append(tid)

    union = sorted(
        set(checked) | set(credential_vulnerable) | set(injection) | set(yaml_mismatch)
    )

    return {
        "categories": {
            "checked": sorted(checked),
            "credential_vulnerable": sorted(credential_vulnerable),
            "injection": sorted(injection),
            "yaml_mismatch": sorted(yaml_mismatch),
        },
        "counts": {
            "checked": len(checked),
            "credential_vulnerable": len(credential_vulnerable),
            "injection": len(injection),
            "yaml_mismatch": len(yaml_mismatch),
            "union": len(union),
            "total_tasks": len(records),
        },
        "evaluation_subset": union,
    }


def main() -> None:
    records = load_audit()
    subset = derive_subset(records)
    OUT_PATH.write_text(json.dumps(subset, indent=2))
    c = subset["counts"]
    print(f"Audit records:            {c['total_tasks']}")
    print(f"Safety-checked tasks:     {c['checked']}")
    print(f"Credential-vulnerable:    {c['credential_vulnerable']}")
    print(f"Injection-defense tasks:  {c['injection']}")
    print(f"YAML-grader mismatches:   {c['yaml_mismatch']}")
    print(f"Evaluation subset (union):{c['union']}")
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

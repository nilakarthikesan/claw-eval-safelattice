# SafeLattice Review and Submission Plan

The working manuscript targets ACM DTRAP. This note records review tasks; venue status, formatting, and submission rules should be checked against the current official instructions before submission.

## Technical review

- Review the security-model mapping and the assumptions under which score aggregation preserves the stated evaluator invariants.
- Examine whether labels and task-specific verifiers capture the claimed violations independently of the scoring implementation.
- Assess false positives, false negatives, narrow obfuscation probes, incomplete task coverage, and uncertainty in live-model measurements.
- Keep synthetic profile ranking changes distinct from the unchanged aggregate live model rankings.
- Check that literature comparisons use compatible metrics and evaluation conditions.

## Artifact review

Reproduce the deterministic corpus measurements and unit/integration checks. Inspect the saved live analysis and its labeling provenance. Confirm file paths, configuration, and commands against the current tree. Record the exact commit and environment for any submission artifact.

## Manuscript review

Preserve upstream credits and applicable licenses. Retain the Use of LLMs method disclosure and the distinction between deterministic checks and LLM judging. Confirm anonymization and repository requirements for the selected venue.

The source is in [overleaf-paper-dtrap/main.tex](overleaf-paper-dtrap/main.tex). The current extension is a research evaluator; benchmark performance and formal properties under a chosen model do not establish general deployed-agent safety.

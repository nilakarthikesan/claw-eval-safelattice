# SafeLattice: External Review Ask + Venue Shortlist

Prepared for discussion with my advisor. Two decisions needed: (1) who gives
us a second technical read, and (2) which venue/deadline we target, since that
determines whether we submit with the current results or wait for the full
five-trial sweep.

## 1. Second-reader ask

The paper's riskiest sections are the ones a general ML reviewer is least
equipped to check and a security reviewer will scrutinize hardest:

- **Formal sections (V: BLP mapping, Formal Properties of SafeLattice).** We
  use the six-step state-machine verification methodology with lemmas for
  lattice well-formedness, a secure-state invariant, and backward
  compatibility. A systems-security colleague who teaches or works with
  BLP/Biba should pressure-test: Is the BLP-to-agent-evaluation mapping sound
  or does it overclaim? Are the proofs stated at the right level of rigor for
  a security venue?
- **Threat model (Sec. III-IV).** Does the write-down/no-write-up framing
  hold up against how an adversarial reviewer would model agent capabilities?

Concrete ask for the advisor: one colleague from the security faculty (ideally
whoever teaches the formal-models material from CS 6238) for a 30-minute read
of Sections IV-V plus the threats-to-validity subsection, before we freeze the
submission.

## 2. Venue shortlist (deadlines verified July 9, 2026)

| Venue | Deadline | Type | Fit |
|---|---|---|---|
| ACM AISec 2026 (at CCS) | **Jul 24, 2026** (firm) | Archival workshop | Strong: AI+security intersection is exactly the topic; but 2 weeks out |
| NeurIPS 2026 workshops (e.g. SEA: Scaling Environments for Agents) | ~Aug 29, 2026 | Non-archival workshop | Good for feedback; non-archival keeps the work submittable elsewhere |
| IEEE SaTML 2027 | **Sep 29, 2026** | Full conference | Best fit: trustworthy ML, verification of algorithms/systems, formal rigor valued; ~12 weeks out |

**Recommendation.** Target **SaTML 2027 (Sep 29)** as the primary venue: the
paper's identity (formal security models applied to ML evaluation) matches its
CFP better than any general ML venue, and the timeline allows (a) the full
five-trial live sweep with human-confirmed labels, (b) the second-reader pass
on the formal sections, and (c) a revision cycle. Optionally submit a
condensed version to a **NeurIPS non-archival workshop (Aug 29)** for early
community feedback at no archival cost. AISec's July 24 deadline is feasible
only if we are comfortable submitting with pilot-scale live results; the
scaled sweep now running weakens that argument for rushing.

## 3. What must be frozen before submission (status)

- Live multi-trial sweep: 1,222 rollouts executed (6 models x 50 tasks x 3-5
  trials, ~$36); the remaining 44 tasks resume with one command
  (`experiments/safelattice/run_live.py`, 12-way parallel).
- Ground-truth labels for live traces: independent-detector proposals plus
  mechanical reference-solution verifiers (`verify_labels.py`) covering the
  six tasks with 86% of detections; 85 verified violations. Binary recall
  0.06 vs SafeLattice 1.0 (F1 0.11 vs 0.81). Spot-check remaining rows in
  `analysis/live_label_proposals.csv` before submission.
- Over-refusal metric separating refusal from incapacity: done
  (`experiments/safelattice/refusal.py`).
- Canary-based obfuscation detection (closes the evasion gap): done
  (`_scan_for_obfuscated_secrets` in `src/claw_eval/graders/safety_enforcer.py`);
  evasion recall on the six-variant probe went 0.33 -> 1.0.
- Related-work freshness pass (2026 follow-ups to TraceSafe-Bench, XGUARD,
  Yu et al.): do in the week before submission so it cannot go stale.

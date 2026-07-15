# SafeLattice: External Review Ask + Venue Shortlist

Prepared for discussion with my advisor. Venue is now decided — **ACM DTRAP**
(see Section 2). The remaining open item is (1) who gives us a second technical
read before we finalize the submission. Because DTRAP is a journal with no
fixed deadline, we can submit on the current results or wait for the full
five-trial sweep without missing a window.

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

## 2. Primary target: ACM DTRAP (Digital Threats: Research and Practice)

**Decision made:** the primary target is **ACM DTRAP**
(https://dl.acm.org/journal/dtrap), a journal, not a conference. This changes
the constraint from *deadline* to *rubric fit and quality* — there is no fixed
submission date, so we submit when the manuscript is ready.

Why DTRAP fits:

- **Extant threats, not laboratory models.** DTRAP explicitly scopes to
  scientifically rigorous work on *real* digital threats. Our paper now leads
  with the OpenClaw CVE chain (ClawJacked, Claw Chain) and the 11 live
  frontier-model credential leaks scored "safe" by the deployed benchmark —
  extant, weaponized threats, exactly the venue's remit.
- **Research–practice bridge.** DTRAP wants work that connects research and
  practice; the tinman-eval / Claw-Eval deployment framing and the graduated
  practitioner tooling argument speak directly to this.
- **Reproducibility.** The trace corpus, dual-scoring harness, live-run
  harness, 119 tests, and released JSON artifacts match DTRAP's
  reproducibility expectations.
- **Advisor precedent.** The advisor's group has an active DTRAP submission
  (DTRAP-2026-0102, "Theia"), which we used as the structural and formatting
  model — de-risks format acceptance.

Manuscript is ready in `overleaf-paper-dtrap/` (acmart, double-anonymous,
26 pages — within DTRAP's ~30-page soft cap). The prior IEEE version remains
untouched in `overleaf-paper/` if a conference fallback is ever needed.

### DTRAP conversion status (done)

- Converted to `acmart` `\documentclass[manuscript,screen,review,anonymous]`
  with `\acmJournal{DTRAP}` and `\setcopyright{none}`, mirroring the Theia
  package.
- Ported all sections, tables, theorems/proofs (dropped the IEEEtran `\proof`
  workaround; acmart's `amsthm` handles it natively).
- Added ACM front matter: CCS concepts (logic and verification, information
  flow control, intelligent agents, security requirements) + keywords.
- Bibliography moved to `references.bib` + `ACM-Reference-Format`; all 53
  cited keys resolve, no bibtex errors.
- Anonymized for double-anonymous review: author block is "Anonymous
  Author(s)"; the repository is referred to as "anonymized ... available to
  reviewers"; the CAPRI-DP self-citation is anonymized in the reference list.
- Added a **"Use of LLMs"** subsection (agents-under-test, LLM-judge in
  grading, and explicit statement that no LLM touches the safety detectors or
  ground truth) and a **Theia-style ground-truth audit table** (Table:
  1,222 rollouts labeled, 3 detectors, 6 mechanical verifiers, 69 confirmed /
  111 overturned, 85 verified violations).
- Reframed abstract + intro to open on extant threats and the deployed-gate
  detection gap.
- Compiles locally with `tectonic` (`cd overleaf-paper-dtrap && tectonic
  main.tex`) → `main.pdf`, 26 pages.

### ScholarOne submission steps (advisor / author only — cannot be automated)

Submission is via ScholarOne Manuscripts (mc.manuscriptcentral.com/dtrap):

1. **Create/confirm ScholarOne account** for the corresponding author and add
   an **ORCID** iD (DTRAP requests it).
2. **Start a new submission**, manuscript type **Research Article** (not Field
   Note / Tool / Editorial).
3. **Upload the anonymized PDF** built from `overleaf-paper-dtrap/` and select
   the double-anonymous option; confirm no author-identifying content (we have
   scrubbed the manuscript, but ScholarOne also asks for a title-page/author
   step kept separate from the blind PDF).
4. **Abstract + keywords + ACM CCS concepts** — paste from the manuscript.
5. **Suggested/opposed reviewers** — advisor to nominate 2–3 suggested
   reviewers (formal-security or agent-safety background) per journal norms.
6. **Cover letter** — one paragraph on extant-threat fit and the
   research–practice contribution; advisor to approve.
7. **Author agreement / conflicts / funding** disclosures as prompted.
8. Keep a **non-anonymous camera-ready** branch ready: at acceptance, switch
   `\documentclass` to the ACM production option, restore the real author
   block and repository URL, and complete the ACM rights (eRights) form
   (`\setcopyright` / `\acmDOI` filled from the rights email).

### Second-reader ask (unchanged, still valuable pre-submission)

See Section 1 — a 30-minute formal-sections read from a systems-security
colleague remains the highest-value pre-submission review, and there is no
deadline pressure forcing us to skip it.

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

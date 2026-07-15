# DTRAP Submission — Message to Prof. Madisetti

Two ready-to-send drafts (email + Canvas), plus how to create the Overleaf
sharing link he asked for.

---

## A. How to create the Overleaf share link (do this first, ~3 min)

1. Zip the manuscript folder: in a terminal,
   `cd /Users/nilakarthikesan/claw-eval && zip -r safelattice-dtrap.zip overleaf-paper-dtrap`
2. Go to https://www.overleaf.com → **New Project** → **Upload Project** →
   select `safelattice-dtrap.zip`.
3. In the project, set the main document to `main.tex` and the compiler to
   **pdfLaTeX** (Menu → Settings). It should build to 26 pages.
4. Click **Share** (top right) → **Turn on link sharing** → copy the
   **"Anyone with this link can view"** URL (use *view*, not *edit*, unless you
   want him to edit).
5. Paste that URL into the email/Canvas message below where it says
   `[OVERLEAF LINK]`.

---

## B. Email

**To:** madisetti.vijay@gmail.com
**Subject:** SafeLattice paper — converted to ACM DTRAP format, Overleaf link inside

Dear Professor Madisetti,

Thank you for sending the DTRAP proof and the Theia LaTeX package — I used both
as the model for this round of work. I have converted our SafeLattice paper into
a DTRAP-ready submission, and the Overleaf link is here:

**[OVERLEAF LINK]**

A summary of what changed and how your resources shaped it:

**Format and structure (from the Theia package).** The paper is now on the ACM
`acmart` class in double-anonymous review mode
(`\documentclass[manuscript,screen,review,anonymous]`, `\acmJournal{DTRAP}`),
using the exact `acmart.cls` and `ACM-Reference-Format.bst` from the package you
shared. I mirrored Theia's section structure — Introduction / Related Work /
Background / Methodology / Results with a **"Use of LLMs" disclosure** and a
**validation-audit table**, then Threats to Validity / Future Work / Conclusion /
reproducibility appendix. Tables now carry the explanatory "Note." footers your
proof uses.

**Reframing to DTRAP's rubric (from the proof + author guidelines).** DTRAP
scopes to *extant* digital threats rather than laboratory models, so the abstract
and introduction now open on real, weaponized attacks — the OpenClaw CVE chain
(ClawJacked CVE-2026-32025; Claw Chain CVE-2026-44112/13/15/18) and our finding
that four of six frontier models leak credential material on an authorized-audit
task while the deployed benchmark scores every leak "safe." SafeLattice is
positioned as the practice-facing fix, with the research–practice bridge
(graduated-severity practitioner tooling vs. binary benchmarks) up front.

**Ground-truth audit table (modeled on Theia's human-validation audit).** Your
proof summarizes annotation validation in a dedicated audit table, so I added the
equivalent for our live evaluation: 1,222 rollouts labeled by three independent
detectors, six task-specific mechanical verifiers confirming 69 violations and
overturning 111 conservative flags, for 85 verified ground-truth violations — and
an explicit statement that no LLM judges any safety outcome.

**Headline results (unchanged in substance, now framed for the venue).** Against
that verified ground truth, the deployed binary gate achieves recall 0.06 versus
SafeLattice's 1.0 (F1 0.11 vs. 0.81), with three severity levels resolved,
bootstrap 95% confidence intervals, and over-refusal at most 0.13 once refusal is
separated from incapacity. The formal sections (BLP mapping, six-step
state-machine proofs, backward-compatibility theorem) carried over intact.

**Logistics.** The manuscript is anonymized for double-anonymous review (author
block, repository reference, and self-citations all scrubbed), compiles cleanly,
and is 26 pages — within DTRAP's ~30-page soft cap. Nothing further is needed in
the LaTeX to submit for review; author names and the repository URL are only
restored at camera-ready if the paper is accepted. The remaining steps
(ScholarOne account, ORCID, Research Article type, suggested reviewers, cover
letter) are in the submission portal, and I've listed them in my notes for
whenever you'd like to proceed.

Could you take a look when you have a chance? I'd especially value your read on
the formal Sections (BLP mapping and the proofs) and whether the extant-threats
framing lands the way DTRAP expects. Happy to meet to walk through it.

Best regards,
Nila Karthikesan

---

## C. Canvas message (shorter backup, in case the email is missed)

Hi Professor Madisetti — I converted the SafeLattice paper into a DTRAP-ready
submission using the DTRAP proof and the Theia LaTeX package you shared. It's now
on the ACM `acmart` template in double-anonymous mode, reframed to lead with
extant threats (the OpenClaw CVE chain and our live finding that four of six
frontier models leak credentials while the deployed benchmark scores them safe),
and I added a Theia-style ground-truth validation-audit table and a "Use of LLMs"
disclosure. It compiles cleanly at 26 pages (within DTRAP's cap) and is
anonymized for review. Overleaf link: [OVERLEAF LINK]. I sent the full details by
email as well — would love your feedback on the formal sections and the framing.
Thank you!

---

## D. Answers to your questions

**Do we need to fill out the placeholders now to submit?**
No. DTRAP uses *double-anonymous* review, so author names must stay OUT of the
manuscript for submission — the "Anonymous Author(s)" block is correct as-is. The
real author block and repository URL are only restored at **camera-ready** (after
acceptance), along with the ACM rights form. The submission-portal items
(ScholarOne account, ORCID, Research Article type, suggested reviewers, cover
letter) are filled in the website, not the LaTeX. The only in-LaTeX cleanup worth
doing before final submission is replacing a few placeholder author names in
`references.bib` (e.g., "Claw-Eval Authors") with the papers' real authors — this
is for citation accuracy, not a blocker, and does not affect anonymity.

**What makes the paper ready now:** ACM `acmart` double-anonymous format; ACM
front matter (CCS concepts + keywords); BibTeX bibliography via
`ACM-Reference-Format`; extant-threats reframing; a "Use of LLMs" subsection; a
ground-truth validation-audit table; anonymization throughout; and a clean
26-page compile within the page cap.

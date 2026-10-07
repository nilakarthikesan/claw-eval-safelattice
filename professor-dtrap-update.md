# SafeLattice Manuscript Review Note

The manuscript source is in [overleaf-paper-dtrap/main.tex](overleaf-paper-dtrap/main.tex). It develops the severity-scoring and information-flow evaluation extension and reports both a constructed trace corpus and saved live-model analysis.

## Review priorities

1. Check the Bell–LaPadula/Biba mappings and the scope of any formal claims against the implemented evaluator.
2. Keep synthetic profile results separate from the six-model live evaluation; live aggregate rankings are unchanged.
3. Review mechanical label verification, false positives, task coverage, and the limits of the six-variant obfuscation probe.
4. Verify source citations and distinguish upstream Claw-Eval work from the extension.
5. Confirm submission formatting, anonymity, and venue requirements against the current official instructions before submission.

The manuscript includes a Use of LLMs subsection describing agents under test and judge involvement. Preserve that methodological disclosure. A successful compile or complete draft does not establish that the research is publication-ready.

Supporting artifacts are in [analysis/](analysis/) and the [experiment guide](experiments/safelattice/README.md).

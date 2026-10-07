# SafeLattice: Agent Safety Evaluation

SafeLattice extends Claw-Eval with graduated severity scoring, information-flow checks, and trace analysis. This research project studies which violations a binary safety score misses and how severity affects evaluation results.

Nila Karthikesan, Georgia Institute of Technology, CS 8903, Summer 2026. Advisor: Prof. Vijay Madisetti.

This repository is a research fork of [Claw-Eval](https://github.com/claw-eval/claw-eval). The upstream authors developed the benchmark tasks, mock services, and core evaluation harness. My contribution is the SafeLattice extension and the associated analysis. Upstream documentation and credits are preserved in [UPSTREAM_README.md](UPSTREAM_README.md).

## Implementation

The extension assigns severity scores of 0.0 (Critical), 0.3 (High), 0.6 (Medium), 0.9 (Low), and 1.0 (Safe). Co-occurring violations receive the minimum score. Bell–LaPadula motivates the confidentiality checks; Biba motivates checks for low-integrity inputs influencing higher-integrity actions. These mappings are evaluation rules, whose usefulness depends on the labels, detectors, and task constraints.

- [safety_taxonomy.py](src/claw_eval/graders/safety_taxonomy.py): severity classes and score aggregation.
- [safety_enforcer.py](src/claw_eval/graders/safety_enforcer.py): trajectory inspection and constraint checks.
- [experiments/safelattice/](experiments/safelattice/): corpus generation, dual scoring, live runs, label verification, and statistics.
- [analysis/](analysis/): committed measurements, reports, label proposals, and trace artifacts.
- [tests/](tests/): unit and integration coverage for the extension.

## Evaluation evidence

The repository contains two distinct evaluations:

1. **Synthetic traces:** 68 deterministic traces generated from 17 scenarios and four simulated behavior profiles. In [safelattice_measurement.json](analysis/safelattice_measurement.json), binary detection has precision 0.786, recall 0.524, and F1 0.629; SafeLattice has precision 0.870, recall 0.952, and F1 0.909. These are measurements on the constructed corpus, rather than live-model performance. Nine false negatives are recovered, while three false positives remain.
2. **Live rollouts:** saved analysis of 1,222 rollouts from six models over 50 tasks, with uneven task coverage and three to five trials. [safelattice_live_detection.json](analysis/safelattice_live_detection.json) reports 85 labeled violations: binary recall 0.059 and F1 0.111; SafeLattice recall 1.0 and F1 0.806, with 41 false positives. Labels use deterministic detectors and task-specific reference-solution checks. The live aggregate model rankings are unchanged between binary and SafeLattice scoring, as recorded in [safelattice_live_stats.json](analysis/safelattice_live_stats.json).

Synthetic profile ranking changes should not be described as changes in the live model rankings. The six-variant obfuscation probe tests a narrow set of encodings; detecting those variants does not establish general resistance to secret exfiltration. Labeling errors, incomplete task coverage, and false positives remain limitations.

## Reproduce the offline measurements

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m experiments.safelattice.trace_corpus
python -m experiments.safelattice.measurement
python -m experiments.safelattice.dual_score
pytest tests/test_safelattice.py tests/test_safelattice_integration.py
```

See [the experiment guide](experiments/safelattice/README.md) for live-run configuration, labeling, and statistical analysis. Live runs require model API access and incur provider charges. The [manuscript](overleaf-paper-dtrap/main.tex) records the methods and limitations in more detail.

## Credits & citation

The underlying benchmark is Claw-Eval by Ye et al. (2026). If you use the benchmark, please cite their work:

```bibtex
@misc{ye2026clawevaltrustworthyevaluationautonomous,
  title={Claw-Eval: Towards Trustworthy Evaluation of Autonomous Agents},
  author={Bowen Ye and Rang Li and Qibin Yang and Yuanxin Liu and Linli Yao and Hanglong Lv and Zhihui Xie and Chenxin An and Lei Li and Lingpeng Kong and Qi Liu and Zhifang Sui and Tong Yang},
  year={2026}, eprint={2604.06132}, archivePrefix={arXiv}, primaryClass={cs.AI},
  url={https://arxiv.org/abs/2604.06132}
}
```

## License

The upstream Claw-Eval code is distributed under the [MIT License](LICENSE). Preserve its copyright and license notice when redistributing that code.

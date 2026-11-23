# Third-revision results, November 2026

This folder holds the compact outputs of the third revision, prepared for submission to *Computers and Electronics in Agriculture*. They come from the same run root as the second revision, `03_result/revision2_20261019`, and a file here supersedes the second-revision file of the same name. No benchmark model is retrained and BiTFI keeps its predictions and its NMAE of 0.0346. Text paths are relative to the project root. [`release_manifest.json`](release_manifest.json) lists the source path, the source and released SHA-256 and the size of every file, together with the files that were left out and why.

## Validation-tuned equal-information LightGBM control

In the second revision, the equal-information LightGBM control (`CovReg-LGBM`) used fixed settings (300 trees, learning rate 0.05, 31 leaves), whereas BiTFI's context length and refinement depth were selected on validation gaps. The third revision selects the LightGBM settings and the refinement depth on validation gaps as well and replaces the control with the selected one (`CovReg-LGBM-tuned`). The decision to tune was made after the test NMAE of the fixed-settings control was known, and the protocol was fixed in [`covreg_tuning/protocol.json`](covreg_tuning/protocol.json) before any tuning inference.

- Validation gaps: the 1,896 refinement-validation gaps (final 20% of 21 training greenhouses) that selected BiTFI's refinement depth, with the target greenhouse removed from the reference bank.
- Candidates: the fixed settings and 39 random draws (seed 42) from a grid of tree number, learning rate, leaf number, minimum leaf size, row and feature subsampling and L2 penalty ([`covreg_tuning/configs.csv`](covreg_tuning/configs.csv)), each with 1, 2, 3 or 5 passes.
- Selection: minimum mean validation NMAE, ties toward fewer passes. Five passes was best for every configuration. The selected configuration (cfg15) has 600 trees, learning rate 0.02, 63 leaves, a minimum of 5 samples per leaf, all rows, 60% of the features per tree and L2 penalty 10. Its validation NMAE was 0.0417, against 0.0434 for the fixed settings and 0.0373 for BiTFI ([`covreg_tuning/selection.json`](covreg_tuning/selection.json), [`covreg_tuning/validation_summary.csv`](covreg_tuning/validation_summary.csv)).
- Test: only the selected combination was run on the 3,421 test masks, once. Run with the fixed settings, the same code reproduced the second-revision predictions exactly, and replacing hidden gap targets with 1e6 left the predictions unchanged ([`covreg_tuning/checks.json`](covreg_tuning/checks.json)).

| Comparison | Reduction (95% CI) | Wilcoxon p | Greenhouses lower |
| :--- | ---: | ---: | ---: |
| BiTFI vs tuned LightGBM control (NMAE 0.0361) | 4.29% (1.48–7.58%) | 0.0137 | 8/10 |
| Tuned LightGBM control vs TimesFM3 (local + cross) | 11.74% (8.25–16.63%) | 0.0020 | 10/10 |
| Tuned vs fixed-settings LightGBM control (0.0372) | 2.80% (2.13–3.68%) | 0.0020 | 10/10 |

The second-revision figures for the fixed-settings control (BiTFI 6.97% lower, 9 of 10 greenhouses) are kept as a sensitivity row. The ridge control is unchanged. `analysis/covreg/` holds the scenario, duration, variable, refinement-pass and stratum summaries with the tuned control in place of the fixed one, and `analysis/indicators/` the VPD and daily-radiation indicators and leave-one-greenhouse-out effect sizes recomputed with it.

## Commands

```bash
export BITFI_RUN_ROOT="$PWD/03_result/revision2_20261019" BITFI_CLEAN=1
for step in --prepare --check --validate --select --test --analyze --tables --indicators; do
  .venv/bin/python 02_model/revision3_covreg_tuning.py $step --workers 40
done
.venv/bin/python 02_model/revision3_covreg_tuning.py --figures   # Figs. 6 and 7 with the tuned control
.venv/bin/python 02_model/revision3_covreg_latex.py              # supplementary table rows
```

`--figures` renders Figs. 6 and 7 through `figures/revision2_figures.py final` with the tuned control under the key `CovReg-LGBM`. The other figure changes of this revision are display options of `figures/revision2_figures.py existing`; without them the command reproduces the earlier figures pixel for pixel.

```bash
# Fig. 3: NMAE axis 0-0.20 (A) and reduction axis 0-100% (B)
.venv/bin/python 02_model/figures/revision2_figures.py existing --only fig3 --no-copy-analysis --aligned-labels --fig3-xmax-a 0.20 --fig3-xmax-b 100
# Fig. 5: common NMAE axis 0-0.25 for all variables
.venv/bin/python 02_model/figures/revision2_figures.py existing --only fig5 --no-copy-analysis --aligned-labels --fig5-xticks 0 0.10 0.20 --fig5-xmax 0.25 --fig5-xminor 0.05
```

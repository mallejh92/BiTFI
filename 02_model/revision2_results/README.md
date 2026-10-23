# Second-revision results, October 2026

This folder holds the compact outputs of the second revision. They come from the run root `03_result/revision2_20261019`, a copy of the complete-gap run `reevaluation_fullgap_20260929` that [`../revision_results/`](../revision_results/) summarizes. The September files in `../revision_results/` are left unchanged, and a file here supersedes the September file of the same name. The stored benchmark predictions of the 21 configurations are reused, and the only new model training is the SAITS seed repeat, which leaves the seed-42 benchmark results unchanged. The 3,421 main masks, the 6,205 scored variable-cases and the scoring function are the same as in September. Text paths are relative to the project root. [`release_manifest.json`](release_manifest.json) lists the source path, the source and released SHA-256 and the size of every file, together with the files that were left out and why.

## What changed

### No-history correction for past-only adapters

The univariate past-only adapters (TimesFM2.5, TimesFM3 and the four AutoGluon models) build each pre-gap context by interpolation and edge filling, which leaves missing positions only when the context holds no valid observation of the target. Those positions were set to zero, the pooled training minimum in min-max space, so the model forecast from a constant series. The corrected adapters (`run_clean_evaluation.no_history_fill`, labeled M4 in the code) skip the forecast in this case and keep the masked-input interpolation, which repeats the first later observation when no earlier one exists.

The correction affects 10 of the 6,205 variable-cases. They lie in six masks (case IDs 2817, 2866, 2915, 2964, 3013 and 3062), all of them 168-h gaps at PF_0025102_01 that start on 3 October 2024, where the record begins with a natural gap that merges with the artificial one. The stored predictions of the six affected configurations were rescored without new inference. The same rule was applied to the matched univariate backbone comparison, whose past-only inputs had the same zero filling.

| Configuration | NMAE before | NMAE after |
| :--- | ---: | ---: |
| TimesFM3 (univariate) | 0.0600 | 0.0573 |
| TimesFM2.5 (univariate) | 0.0614 | 0.0587 |
| LightGBM | 0.0611 | 0.0584 |
| Random forest | 0.0673 | 0.0646 |
| DeepAR | 0.0720 | 0.0693 |
| PatchTST | 0.0726 | 0.0699 |
| Chronos 2, matched univariate comparison | 0.0595 | 0.0568 |
| TimesFM2.5, matched univariate comparison | 0.0614 | 0.0587 |
| TimesFM3, matched univariate comparison | 0.0600 | 0.0573 |

The BiTFI reduction relative to LightGBM changed from 43.41% to 40.80% (95% CI 35.40–46.21%), and the reduction of TimesFM3 (local) relative to TimesFM3 (univariate) changed from 17.94% to 14.08% (95% CI 11.09–17.00%). In the matched univariate comparison, TimesFM3 differed from Chronos 2 by −0.88% instead of −0.84% and from TimesFM2.5 by 2.45% instead of 2.34%. All p values and Holm-adjusted p values are unchanged. Among the 21 configurations, random forest moved from rank 16 to 14, spatial ridge from 14 to 15 and seasonal naive from 15 to 16. In the equal-case sensitivity, PatchTST moved from rank 19 to 18 and spatial ridge from 18 to 19. The eight representative configurations keep their order, BiTFI keeps its NMAE of 0.0346 and its 15.52% reduction relative to TimesFM3 (local + cross), and the values of the other 15 configurations are unchanged.

### BiTFI information-source and reference controls

Six variants of BiTFI (TimesFM3, 1,900-h context, five synchronous refinement passes) change only the covariates used during refinement. `BiTFI-local` drops the references, `BiTFI-cross` drops the other local sensors and keeps the calendar features and references, `BiTFI-calendar` keeps the calendar features alone, `BiTFI-k1` keeps the top reference, `BiTFI-random3` draws three eligible references at random, and `BiTFI-exclTop5` removes the five training greenhouses that supplied most reference slots before taking the top three. Relative to BiTFI (NMAE 0.0346), NMAE increased by 25.47% (`BiTFI-local`, 0.0434), 6.75% (`BiTFI-cross`, 0.0369), 44.13% (`BiTFI-calendar`, 0.0499), 5.94% (`BiTFI-k1`, 0.0367), 1.44% (`BiTFI-random3`, 0.0351) and 2.97% (`BiTFI-exclTop5`, 0.0356). The last two increases were not significant after Holm adjustment over the six variants (adjusted p = 0.2617).

### Equal-information regression controls

`CovReg-LGBM` and `CovReg-Ridge` receive the gap jobs, the two-sided context windows, the covariate block (other local sensors, calendar features and three references) and the five synchronous refinement passes of BiTFI, but a LightGBM or ridge regression fitted for each mask and target replaces the frozen backbone. Pass 0 is masked-input linear interpolation, and neither regression was tuned. Their NMAEs were 0.0372 and 0.0498. BiTFI reduced them by 6.97% (95% CI 3.88–10.58%, Wilcoxon p = 0.0059, lower in 9 of 10 greenhouses) and by 30.59%.

### SAITS training seeds

SAITS (local) and SAITS (local + cross) were retrained with seeds 43–46 on the fixed windows, reference features and artificial corruptions, in addition to the original seed 42. SAITS (local + cross) NMAE ranged from 0.0509 to 0.0527 (mean 0.0519) and SAITS (local) NMAE from 0.0546 to 0.0572 (mean 0.0562). The BiTFI reduction relative to SAITS (local + cross) ranged from 32.08% to 34.41%, and all 10 greenhouses favored BiTFI at every seed (Holm-adjusted p = 0.0137). The reference gain within SAITS stayed positive (6.54–9.83%), while its Wilcoxon p ranged from 0.049 to 0.160. Retraining with seed 42 reproduced the original weights and scores exactly.

### VPD and daily-radiation indicators and leave-one-greenhouse-out effect sizes

Vapor pressure deficit (VPD) is computed from indoor temperature and RH with the Tetens form of the saturation vapor pressure (FAO-56 Eqs. 11 and 54) for Scenario B and C gaps, which mask both variables. Daily outdoor radiation sums the hourly irradiance of calendar days that lie entirely inside a masked radiation interval (Scenarios A and C). Both indicators are recomputed from the stored predictions of the main masks and use the greenhouse weighting and paired tests of the main analysis. The seven representative comparisons form one Holm family per indicator, scope and output variant, and the two regression controls form a separate two-comparison family (`covreg_holm2`).

BiTFI had an hourly VPD MAE of 0.0641 kPa in Scenario B and 0.0694 kPa in Scenario C, and a daily-radiation absolute error of 0.995 MJ m⁻² d⁻¹ in Scenario A and 1.385 MJ m⁻² d⁻¹ in Scenario C. For single-sensor radiation gaps, its daily-sum error did not differ significantly from TimesFM3 (local + cross) (5.53% lower, Holm-adjusted p = 0.32), and `CovReg-LGBM` had the lower mean error (BiTFI 5.31% higher, p = 0.49). Removing any one test greenhouse kept the main NMAE reduction relative to TimesFM3 (local + cross) between 14.02% and 16.80% (15.52% with all ten), and BiTFI remained lower in every leave-one-out subset for all seven representative comparisons. The reduction relative to `CovReg-LGBM` ranged from 6.10% to 8.04%.

### Diagnosis of the largest indicator error

The largest BiTFI VPD error occurred in mask 2727 (PF_0025101_01, labeled Test 06 in the manuscript, Scenario C, 168 h from 10 July 2025 20:00). This mask was rerun with BiTFI, `BiTFI-cross`, `BiTFI-local`, forward-only BiTFI and four BiTFI runs that each drop one local sensor from the indoor-temperature calls, and the forward, backward and fused estimates of every pass were stored. The reruns differed from the stored predictions by less than 10⁻⁶ in normalized units. The BiTFI VPD MAE of this mask moved from 0.151 kPa after initialization to 0.508, 0.317, 0.495, 0.342 and 0.513 kPa over the five passes, whereas `BiTFI-cross` stayed at 0.085 kPa in every pass and forward-only BiTFI fell steadily to 0.076 kPa. A post hoc screen then classified every scored variable-case by the observed run of its variable directly after the artificial gap. In the 140 variable-cases (83 masks, 9 greenhouses) whose observed run was shorter than 48 h and followed by missing values, the unweighted mean NMAE was 0.0381 for BiTFI and 0.0362 for forward-only BiTFI. In the other 6,065 variable-cases it was 0.0304 and 0.0355.

## Files

| Folder | Content | Written by |
| :--- | :--- | :--- |
| `analysis/` | The 21-configuration benchmark tables after the correction. These are the summaries by scenario, gap duration, variable, season and physical unit, the greenhouse scores, the paired-test families, the equal-case and constant-exclusion sensitivities, the refinement-depth, matched B/C and residual-strata tables, the reporting sensitivity, the A6000 timing and the source hashes | `revision2_m4_fix.py reaggregate`, which runs `hourly_analysis.py` without rewriting evaluation outputs |
| `source_data/` | The eight-configuration summary (`main_summary.csv`), its greenhouse scores and paired tests, and the matched univariate backbone summary, re-rendered after the correction | `figures/revision2_figures.py existing` (overall-performance builder of `figures/build_prism_figures.py`) and `existing --with-s6` (`figures/FigureS7_univariate_backbones.py`) |
| `sensitivity/` | Percentile-range normalization, common-bound clipping, quality and reference summaries recomputed on the corrected run root | `verified_sensitivity_analysis.py --root <run root> --output <run root>/verification_20260929/sensitivity_m4` |
| `revision2/bitfi_variants_*` | BiTFI information-source and reference controls by scenario, duration, variable and greenhouse, paired tests, reference use and timing | `revision2_bitfi_variants_analysis.py` |
| `revision2/covreg_*` | Equal-information regression controls, their strata, refinement passes, fallbacks and input hashes | `revision2_covreg.py --analyze` |
| `revision2/saits_seeds_*` | SAITS training-seed repeats, seed spread, paired tests, ranks and reproduction checks | `revision2_saits_seeds.py analyze` |
| `revision2/indicators_*`, `revision2/loo_effect_sizes*` | VPD and daily-radiation indicators and leave-one-greenhouse-out effect sizes. Files ending in `_covreg` hold the regression controls in their own Holm family | `revision2_indicators.py` |
| `failure_case_2727/` | Diagnosis of mask 2727 (`summary.csv`, `complete.json`) and the post-gap run screen (`post_gap_run_screen.csv`) | `revision2_failure_case.py` and `revision2_failure_case.py --screen` |

Several of these files replace September tables with pre-correction values.

- `source_data/main_summary.csv`, `main_greenhouse_scores.csv`, `paired_greenhouse_tests.csv` and `univariate_backbone_summary.csv` replace the files of the same name in `../revision_results/`.
- In `analysis/`, the `all_*` tables, `main_paired.csv`, `component_paired.csv`, `univariate_paired.csv`, `equal_case_summary.csv`, `equal_case_sites.csv`, `constant_exclusion_summary.csv` and `source_manifest.json` replace their September versions. The other analysis tables are byte-identical to the September files, and `constant_case_flags.csv`, `matched_availability_cases.csv` and `reference_availability_counts.csv` are included here for the first time.
- `sensitivity/` replaces `../revision_results/sensitivity/`. The September `rank_audit.json` has no corrected counterpart, and the rank columns of `sensitivity/normalization_summary.csv` give the corrected ranks.
- The September verification records in `../revision_results/` (`verified_autogluon_summary.*`, `verified_autogluon/`, `fullgap_verification/`, `verification_fullgap_20260929/` and `publication_validation.json`) were not regenerated. They document the September runs and checks and keep the September values of the six affected configurations.

## Reproducing the files

Run from the repository root. The revision-2 drivers expect `BITFI_RUN_ROOT` to name a copy of the completed full-gap run root, with its model folder linked read-only. They also read stored outputs of the original root, `03_result/reevaluation_fullgap_20260929`, and refuse to write into it. The Revision 2 section of [`../README.md`](../README.md) describes the inputs, outputs and environments of each script.

```bash
export BITFI_RUN_ROOT="$PWD/03_result/revision2_20261019"
export BITFI_CLEAN=1 JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1
export HF_HOME="$PWD/03_result/model_cache/huggingface"

# No-history correction, re-aggregation and sensitivity (standard environment, CPU)
for step in audit verify rescore univariate reaggregate; do
  .venv/bin/python 02_model/revision2_m4_fix.py "$step"
done
(cd 02_model && ../.venv/bin/python verified_sensitivity_analysis.py --root "$BITFI_RUN_ROOT" --output "$BITFI_RUN_ROOT/verification_20260929/sensitivity_m4")
.venv/bin/python 02_model/revision2_m4_fix.py compare --save "$BITFI_RUN_ROOT/m4_fix/changed_values_analysis.csv"

# BiTFI information-source and reference controls (TimesFM3 environment and one GPU, merge on CPU)
.venv/bin/python 02_model/revision2_bitfi_reference_audit.py
.venv_tfm3/bin/python 02_model/revision2_bitfi_variants.py --repro
for v in BiTFI-local BiTFI-cross BiTFI-calendar BiTFI-k1 BiTFI-random3 BiTFI-exclTop5; do
  .venv_tfm3/bin/python 02_model/revision2_bitfi_variants.py --variant "$v" --smoke
  for shard in 0 1; do
    .venv_tfm3/bin/python 02_model/revision2_bitfi_variants.py --variant "$v" --shard "$shard" --shards 2
  done
  CUDA_VISIBLE_DEVICES= .venv/bin/python 02_model/revision2_bitfi_variants.py --variant "$v" --merge
done
.venv/bin/python 02_model/revision2_bitfi_variants_analysis.py

# Equal-information regression controls (standard environment, CPU; block check in the TimesFM3 environment)
for m in CovReg-LGBM CovReg-Ridge; do
  .venv/bin/python 02_model/revision2_covreg.py --model "$m" --smoke
done
.venv_tfm3/bin/python 02_model/revision2_covreg.py --check-block
for m in CovReg-LGBM CovReg-Ridge; do
  .venv/bin/python 02_model/revision2_covreg.py --model "$m" --workers 20
done
.venv/bin/python 02_model/revision2_covreg.py --analyze

# SAITS training seeds (standard environment, one GPU for run and smoke)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python 02_model/revision2_saits_seeds.py run
CUDA_VISIBLE_DEVICES=0 .venv/bin/python 02_model/revision2_saits_seeds.py smoke
.venv/bin/python 02_model/revision2_saits_seeds.py analyze

# Indicators and leave-one-greenhouse-out effect sizes (standard environment, CPU, after the correction)
.venv/bin/python 02_model/revision2_indicators.py

# Diagnosis of mask 2727 (TimesFM3 environment, one GPU) and post-gap screen (standard environment, CPU)
CUDA_VISIBLE_DEVICES=0 .venv_tfm3/bin/python 02_model/revision2_failure_case.py
.venv/bin/python 02_model/revision2_failure_case.py --screen
```

`revision2_m4_fix.py locate` looks up the affected manuscript strings and needs the private manuscript sources, so it is not part of this sequence. The variants were originally queued with `revision2_bitfi_queue.py --gpu <1|2> --jobs ...`, which runs the same calls one at a time on one GPU.

## Figures

The figure script runs on CPU without model inference and reads the tables of one run root.

```bash
.venv/bin/python 02_model/figures/revision2_figures.py existing --with-s6 --aligned-labels
.venv/bin/python 02_model/figures/revision2_figures.py final
.venv/bin/python 02_model/figures/revision2_figures.py new --only s13 s14
```

`existing` re-renders the result figures that contain rescored configurations (overall performance, gap robustness, variable performance, the extended comparison, seasonal performance, the available-information comparison and, with `--with-s6`, the univariate backbone comparison). `--aligned-labels` places each panel letter on the baseline of its panel title. `final` writes the BiTFI information-source and reference figure, the agricultural-indicator figure, the reconstruction example, the controls by scenario and by variable, the equal-information controls by gap duration and the failure-case figure. The reconstruction example is a byte copy of an author-drawn PDF from the private manuscript folder, so outside that setting run `final --only fig6 fig7 s11 s12 s15 s16`. `new --only s13 s14` writes the SAITS seed-repeat and greenhouse-dependence figures. The keys after `--only` name builders in the script and do not follow the manuscript numbering.

Since 23 October 2026 the manuscript supplement keeps three figures. Fig. S1 (`FigureS1_Context_sensitivity`) is written by the context-selection builder `figures/FigureS2_validation_context.py`, Fig. S2 (`FigureS2_Univariate_backbone_comparison`) by `existing --with-s6` and Fig. S3 (`FigureS3_Failure_case`) by `final --only s16`. The other supplementary builders write figures that are no longer part of the supplement, and their output names keep the numbers of 20 October 2026.

Main figures go to `<run root>/figures/`, supplementary figures to `<run root>/figures/Supplementary/`, figure source tables to `<run root>/figures/source_data/` and render logs to `<run root>/figures/revision2/`. The `new` builders write to `<run root>/figures/revision2/` and its `Supplementary/` subfolder. The script refuses to write into the manuscript folders or the original run root. The default font is Liberation Sans, and `pdftoppm` and `pdftotext` (poppler-utils) must be on the path for the PNG previews and checks.

## Files not included

| File | Reason |
| :--- | :--- |
| `analysis/comparison_results.csv`, `analysis/refinement_cases.csv`, `analysis/residual_strata_cases.csv` | Case-level tables larger than 1 MB |
| `analysis/additional_*.csv` (4 files) | The additional-site analysis lies outside the current publication scope and was not regenerated after the correction. The September copies in `../revision_results/` are byte-identical |
| `analysis/early_physical_MAE.csv` | Table left from an earlier aggregation stage. The current scripts do not write it, and the re-aggregation did not update it |
| `revision2/indicators_vpd_cases.csv`, `revision2/indicators_vpd_days.csv`, `revision2/indicators_radiation_cases.csv`, `revision2/indicators_radiation_days.csv`, `revision2/indicators_vpd_days_covreg.csv`, `revision2/indicators_radiation_days_covreg.csv` | Case- and day-level indicator tables larger than 1 MB |
| `revision2/covreg_analysis.log` | Console log, not a result table |
| `failure_case_2727/components.npz` | Per-pass forecast arrays |
| Other files of `sensitivity_m4/` | Case-, site- and run-level diagnostics, the normalizer table and manuscript text fragments, which the September sensitivity release also left out |

Model weights, fitted checkpoints, individual prediction arrays, the case-level audit records of the correction and the evaluation folders of the new controls are not included either.

## Notes

- `failure_case_2727/complete.json` records the SHA-256 of `revision2_failure_case.py` before the `--screen` option was added. The diagnosis path itself did not change. The other code hashes recorded in these files match the released scripts.
- `revision2/indicators_provenance.json` embeds the completion record of the correction. Its `sha256_before` entry is the September version of `run_clean_evaluation.py`, and `sha256_after` is the released version.
- The two `revision2/*_cases_covreg.csv` files are case-level tables below 1 MB and are therefore included.

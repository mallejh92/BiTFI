# Verified hourly results, 29 September 2026

These tables accompany the current 21-configuration greenhouse reconstruction benchmark. They summarize run `revision_verified_20260929`: 34 complete hourly-grid sites (24 training, 10 main test), 3,421 main masking cases and 6,205 scored variable-cases. Nine additional sites are evaluated separately. Each greenhouse receives equal overall weight after observation-weighted aggregation within that greenhouse.

## What was executed

Six existing configurations were refitted and reevaluated on the unchanged main masks: `AG-LightGBM`, `AG-RandomForest`, `AG-DeepAR`, `AG-PatchTST`, sensors-only `SAITS`, and `MOMENT-FT`. AutoGluon now receives the records' actual hourly timestamps throughout fitting and recursive prediction. The training budget is a requested 600 s per model, a soft library deadline. Sensors-only SAITS uses a 300-epoch cap and patience 10; its validation-selected epoch is 70. MOMENT tunes only the 8,200-parameter reconstruction head with a frozen encoder, five learning rates, a final 1,000-epoch cap and patience 20; validation selects learning rate 0.001 and epoch 112. Checkpoint selection was frozen before revised test evaluation.

Outputs for the other 15 primary configurations were reused from `reevaluation_hourly_20260927`; this is not a full 21-configuration rerun. In particular, the main 24-channel SAITS local and local + cross controls and the selected five-pass BiTFI predictions are unchanged. The direct five-pass production path was verified against stored outputs, hidden-target perturbations, serial inference and reordered batches at the original numerical tolerances. Matrix precision is set to `highest` after TimesFM3 construction. Bitwise equality is not claimed across arbitrary hardware and batch grouping.

No new model family was added. The supplementary full-gap-call control uses the existing TimesFM3 local + cross configuration, holds its auxiliary initialization fixed, and changes only covariate-target calls from rolling to complete-gap prediction. Percentile-range normalization and common-bound clipping are sensitivity analyses of saved predictions; primary predictions are not replaced by clipped values.

## Files

- `all_summary.csv` contains all 21 primary configurations; `main_summary.csv` contains the eight representative configurations. The other `all_*.csv` tables give variable, gap, scenario, season and physical-unit results.
- `main_paired.csv`, `component_paired.csv`, `refinement_paired.csv`, `additional_paired.csv` and `univariate_paired.csv` report the five separately Holm-adjusted test families. `paired_greenhouse_tests.csv` supplies the figure-oriented main comparisons.
- `refinement_*.csv` includes Step 1 and tested pass counts, including scenario-specific scores. `equal_case_*.csv`, `scenario_duration_weights.csv`, `constant_exclusion_summary.csv` and `residual_strata_*.csv` provide the weighting and descriptive quality analyses.
- `context_validation/` and `refinement_validation/` report training-site validation selection. The 1,900-h context and five passes are selected within finite tested ranges, not established global optima.
- `verified_autogluon/`, `verified_autogluon_summary.*` and `verified_training/` document the six reruns, timing, validation-only choices and checks. Interrupted attempts are not successful model results.
- `production/` records selected-depth numerical checks; `single_call/` reports the complete-gap-call control; `sensitivity/` reports percentile normalization, clipping, reference availability and data-quality summaries.
- `figure6_panel_metrics.csv` contains the fixed illustration's scores, including the updated MOMENT head. The illustrated interval was not reselected.
- `publication_validation.json` is the completed independent verification report for predictions, scoring, table counts and reporting consistency. Its prediction checks cover all 21 configurations; absence of a top-level Boolean reflects the report schema, not an incomplete run.
- `source_manifest.json` retains source protocol and result hashes. `release_manifest.json` records source and released hashes for this compact payload. Only absolute local project-root prefixes in text paths were removed for portability; scientific values are unchanged.

## Main outcome and limits

BiTFI's primary NMAE is 0.034596565, compared with 0.041657620 for TimesFM3 local + cross (16.95% relative reduction). The complete-gap-call control scores 0.041601672, yielding a 16.84% reduction for BiTFI. The main eight-configuration ranking is unchanged under percentile-range normalization, with a 16.48% reduction for BiTFI over TimesFM3 local + cross. These controls do not establish that every fixed engineering threshold is optimal.

See `../README.md` for a fresh reproduction. The standard hourly entry points recompute all required training and evaluation inputs. The revision-specific `run_verified_training.py` and `run_verified_autogluon.py` drivers instead document the isolated update of an existing completed experiment and are not fresh-install commands. Downloaded model weights, fitted checkpoints, individual prediction arrays and private manuscript files are excluded from this release; the compact aggregate tables here do not replace the full local experiment.

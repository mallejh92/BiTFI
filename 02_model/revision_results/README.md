# Complete-gap benchmark results, 29 September 2026

This compact release summarizes `reevaluation_fullgap_20260929`: 21 primary configurations, 3,421 main masks and 6,205 scored variable-cases across ten test greenhouses. Training and selection use the separate 24-site pool; nine additional sites are evaluated separately. Scores weight hours within each greenhouse and greenhouses equally.

## Current forecasting protocol

TimesFM2.5 and all four TimesFM3 benchmark adapters request each complete missing span in one native forecast call. The auxiliary univariate initialization used by the covariate adapters follows the same rule. External 128-hour rolling is disabled; internal pretrained decoding is unchanged. The 1,900-hour context remains the training-validation selection reported in Figure S2, which already used complete-gap calls.

Five frozen TimesFM configurations were reevaluated; the other 16 primary outputs were retained from `revision_verified_20260929`. No fitting was performed for this change. The prior run's four AutoGluon models, sensors-only SAITS and MOMENT-head training records remain applicable. The main BiTFI five-pass predictions and the existing S2/S6, matched B/C, refinement and additional-site controls are unchanged. The fixed illustration reexports only TimesFM3 (local + cross).

Where an artificial gap joins a natural gap beginning at the first record, auxiliary initialization uses reversed post-gap observations. Its predictions are restored to chronological order before insertion. The ten leading variable-cases in six masks were explicitly checked; affected covariate configurations were restarted after this boundary alignment fix. Interrupted partial outputs are excluded. Other model families and BiTFI retain their independent prediction paths.

## Results and files

BiTFI NMAE is 0.034596565, compared with 0.040952758 for TimesFM3 (local + cross): a 15.52% relative reduction. Interpret this as a comparison of the complete reconstruction procedures. Local and cross inputs, directions and refinement are described separately in the component comparisons.

- `all_summary.csv` lists 21 settings; `main_summary.csv` lists 8 representative settings. Other aggregate CSV files cover variables, durations, scenarios, seasons and physical-unit errors.
- Five Holm families are in `main_paired.csv`, `component_paired.csv`, `refinement_paired.csv`, `additional_paired.csv` and `univariate_paired.csv`.
- `context_validation/` and `refinement_validation/` retain training-site parameter selection. Neither maximum tested candidate establishes a global optimum.
- `fullgap_verification/` and `verification_fullgap_20260929/` document whole-gap calls, boundary alignment, hidden-target perturbations, execution order and score verification. Publication validation independently checks all 21 stored prediction sets.
- `sensitivity/` contains percentile-range normalization, physical-bound clipping and quality/reference summaries. Primary predictions are not replaced by clipping.
- `verified_autogluon/` and `verified_training/` record previously completed fitting of the retained comparison models; `production/` retains the selected-depth BiTFI verification.
- `source_manifest.json` records result hashes; `release_manifest.json` maps source and exported hashes. Project-root and home-cache prefixes are normalized in text paths for portability.

The earlier target-only single-call control is retained in git history; it is not the current primary protocol because it kept 128-hour auxiliary initialization. Weights, individual prediction arrays, full case-level evaluation tables and private manuscript files are excluded. See `../README.md` for fresh reproduction. The revision-specific drivers require their documented existing inputs; this compact summary payload does not replace the full local experiment.

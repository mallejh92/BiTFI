# TimesFM2.5 complete-gap reevaluation — complete

The primary TimesFM2.5 configuration now requests each artificial gap in one native forecast call. This removes the outer 128-h split and reinsertion loop. The pretrained weights, masks, target-only past context, 1,900-h input limit, input filling and scalers are unchanged. The model's internal 128-position decoding patches remain part of its native implementation.

| Measure | Previous external 128-h rolling | Complete-gap call |
|---|---:|---:|
| Mean greenhouse NMAE | 0.061414022241 | 0.061416470327 |

All 3,421 mask cases, 6,205 unique variable-cases and 345,324 scored hourly occurrences were evaluated on NVIDIA RTX A6000 (GPU 1). Main evaluation elapsed 510.98 seconds. No cached supplemental predictions were substituted for this rerun. The new main results remain 0.0614 at four decimal places.

## Validation

- All 15 scenario-duration combinations passed hidden-target poisoning and A–B–A execution-order checks.
- All 15 combinations issued exactly one whole-gap call per masked variable and passed serial/batch and reversed-batch checks (maximum difference zero).
- All 3,421 stored prediction files were independently inverse-scaled and scored against raw withheld observations. Maximum MAE/NMAE CSV difference was below 0.0000005, consistent with six-decimal CSV rounding. Independent overall NMAE: 0.061416469841.
- 6, 12, 24 and 72-h predictions are bitwise identical to the preceding main results; only 168-h predictions changed.
- The 1,900-h contexts, masks and preprocessing are unchanged. All 6,205 main context vectors exactly match the existing matched-univariate supplemental inputs.
- The largest main-versus-S6 scaled prediction difference (0.000533462) was reproduced by the recorded batch configurations. Replaying the original S6 batch reproduced its stored predictions exactly, and the new main configuration reproduced its own saved prediction exactly. S6 therefore remains reusable.

## Why Figure S2 remains unchanged

The validation-only context sweep already used one complete-gap native call for every tested length. Its cases come exclusively from the final 20% of training-greenhouse records, with no test greenhouses. All three backbones selected 1,900 h among the seven tested values. Input files, outputs and selected-context records are hash-identical to the preceding run.

## Native capacity

The TimesFM2.5 library's output capacity is compiled to 256 h (an exact multiple of its 128-position output patch), supporting all tested gaps through 168 h. The retained continuous quantile head supports at most 1,024 h; no benchmark case approaches that limit. Main requests use the actual gap length, not the compiled capacity.

## Evidence and reproducibility

- `call_and_batch_checks.json`: all 15 whole-gap call and batching checks.
- `stored_prediction_validation.json`: independent physical-unit scoring, case coverage and input/prediction hashes.
- `s6_batch_numerical_check.json`: exact recreation of the largest cross-configuration numerical difference.
- `unchanged_context_and_s6.json`: selection/input hashes and validation-site checks.
- `model_provenance.json`: installed library and model-weight SHA-256, model snapshot revision.
- `gap_summary_before_after.csv`: duration-specific before/after summaries.
- `02_model/verify_fullgap_timesfm25.py`: `smoke`, `stored` and `s6` audit commands, run with `BITFI_CLEAN=1` and `BITFI_RUN_ROOT` set to this run.

# BiTFI implementation and evaluation

This code accompanies *Bidirectional imputation of missing greenhouse sensor data using frozen time-series foundation models*. The computational protocol uses a complete hourly grid, in which absent timestamps remain missing and one input position is one elapsed hour. Duplicate or off-hour timestamps raise an error. The released site split is fixed; 24 complete sites supply training data, ten complete sites supply main evaluation, and nine incomplete sites are reserved for additional-site evaluation.

## Environments

Use Python 3.12.3 and two environments. The pinned snapshots in `environments/` describe the actual execution environments, including dependencies not directly required by every script. GPU drivers and model downloads must be supplied separately. They have not been tested by installing on a fresh machine.

```bash
python3.12 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r 02_model/environments/standard.requirements.txt
python3.12 -m venv .venv_tfm3
.venv_tfm3/bin/pip install -r 02_model/environments/timesfm3.requirements.txt
git clone https://github.com/WenjieDu/SAITS.git third_party/SAITS
git -C third_party/SAITS checkout 660b87f19c1277065f314f24134f646229e89ca9
```

The standard environment uses PyTorch 2.6.0/CUDA 12.4, Transformers 4.57.6, Chronos 2.2.2 and AutoGluon-TimeSeries 1.5.0. The TimesFM3 environment uses TimesFM 3.0.1 and PyTorch 2.14.0/CUDA 13.0. Select a compatible GPU with `CUDA_VISIBLE_DEVICES`. RTX A6000 timings are separated from computation performed on RTX PRO 6000 Blackwell hardware.

The exact pretrained revisions and required files are listed in `environments/checkpoints.json`. Populate the project cache once (network access required), then keep it offline for an evaluation:

```bash
.venv/bin/python 02_model/prepare_foundation_weights.py
export HF_HUB_OFFLINE=1
```

## Prepare a fresh experiment

Place the Smart Farm Korea records in `01_data/` as described in [`../01_data/README.md`](../01_data/README.md), then run from the repository root. `BITFI_RUN_ROOT` isolates data, fitted models and predictions; choose a new directory for a changed protocol. The initializer refuses to overwrite an existing resolved split.

```bash
export BITFI_RUN_ROOT="$PWD/03_result/reproduction_fullgap_20260929"
export BITFI_CLEAN=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python 02_model/initialize_public_paths.py
.venv/bin/python 02_model/clean_protocol.py
.venv/bin/python 02_model/hourly_revision.py windows
.venv/bin/python 02_model/hourly_revision.py context-prepare
.venv/bin/python 02_model/hourly_revision.py saits-prepare
.venv/bin/python 02_model/hourly_quality_audit.py
.venv/bin/python 02_model/hourly_reference_audit.py
.venv/bin/python -m unittest discover -s 02_model -p test_hourly_protocol.py
```

The current protocol yields 3,421 main masks, 6,205 scored variable-cases, 662 training windows and 41 validation windows. No interpolation precedes masking or scaler fitting. Each variable's scaler is fitted to the pooled first 80% of training-site hourly spans. Test NMAE uses the full-site range after physical screening for scoring only.

## Select settings and fit comparison models

```bash
.venv/bin/python 02_model/hourly_revision.py context --model Chronos2
.venv/bin/python 02_model/hourly_revision.py context --model TimesFM2.5
.venv_tfm3/bin/python 02_model/hourly_revision.py context --model TimesFM3.0
.venv/bin/python 02_model/hourly_revision.py context-select
.venv/bin/python 02_model/hourly_revision.py auxiliary --aux-action prepare
for model in SAITS MOMENT-FT SAITS-matched-local SAITS-spatial; do
  .venv/bin/python 02_model/hourly_revision.py train --model "$model"
done
.venv/bin/python 02_model/hourly_revision.py saits-evaluate
.venv/bin/python 02_model/hourly_revision.py depth-prepare
for shard in 0 1; do
  .venv_tfm3/bin/python 02_model/hourly_revision.py depth --shard "$shard" --shards 2
done
.venv/bin/python 02_model/hourly_revision.py depth-select --shards 2
```

The current training entry point uses a 300-epoch cap and patience 10 for sensors-only SAITS, and a 1,000-epoch cap, patience 20 and learning rates 1e-4, 3e-4, 1e-3, 3e-3 and 1e-2 for the MOMENT head. The selected checkpoints were epochs 70 and 112, respectively; both stopped early. The two 24-channel SAITS controls retain their own matched stopping protocol. The four AutoGluon models, sensors-only SAITS and MOMENT head were fitted in the verified training update. The subsequent full-gap evaluation recomputes the five TimesFM2.5/TimesFM3 input configurations with frozen weights and retains the other 16 primary configurations. Both updates record rerun and reuse scope separately. A fresh reproduction following the commands here instead computes every result from its inputs.

Context selection uses 1,390 single-variable gaps from 21 training-site validation tails, testing 96, 168, 336, 720, 1080, 1440 and 1900 h. Each backbone selected 1900 h, held fixed across its input configurations. Refinement selection uses 1,896 validation masks from 21 training sites and selects five passes among 1, 2, 3 and 5. These maxima are selected within the tested ranges, not established global optima.

`run_clean_evaluation.build` and the hourly entry points read the saved context and refinement selections. The generic batch path runs the selected five synchronous refinement passes. The low-level BiTFI class still accepts an explicit `refinements` argument; use the selected run settings when calling it directly. TimesFM3 loaders change matrix precision, so the inference entry points set `torch.set_float32_matmul_precision('highest')` after model loading.

## Evaluate the 21 configurations

TimesFM2.5 and all TimesFM3 benchmark adapters request each complete gap in one native forecast call, including the auxiliary univariate initialization used by covariate adapters. The default `forecasting_mode="full_gap"` removes external 128-hour rolling; internal pretrained decoding is unchanged. An explicit `"rolling"` option is retained for historical controls. The context is still the 1,900-hour value selected in Fig. S1. For auxiliary gaps beginning at the start of a record, reversed post-gap context is used and the forecast is reversed back to chronological order before insertion. This fallback does not make the ordinary univariate benchmark use future target observations.

Every ordinary evaluation is preceded by its 15-case perturbation/order smoke check. AutoGluon trains its model when first requested, with actual training, validation and context timestamps preserved throughout rolling prediction. The requested training budget is 600 s per model; the library may finish an active fit after this soft deadline. Do not run two workers for the same model directory concurrently.

```bash
for model in LI SeasonalNaive Spatial-Ridge AG-LightGBM AG-RandomForest AG-DeepAR AG-PatchTST SAITS MOMENT MOMENT-FT Chronos2 TimesFM2.5; do
  .venv/bin/python 02_model/hourly_revision.py evaluate --model "$model" --smoke
  .venv/bin/python 02_model/hourly_revision.py evaluate --model "$model"
done
for model in TimesFM3.0 TimesFM3.0-MV TimesFM3.0-COV TimesFM3.0-COV-SPA; do
  .venv_tfm3/bin/python 02_model/hourly_revision.py evaluate --model "$model" --smoke
  .venv_tfm3/bin/python 02_model/hourly_revision.py evaluate --model "$model"
done
.venv_tfm3/bin/python 02_model/hourly_revision.py refinement --smoke
for shard in 0 1 2; do
  .venv_tfm3/bin/python 02_model/hourly_revision.py refinement --shard "$shard" --shards 3
done
.venv_tfm3/bin/python 02_model/hourly_revision.py controls --model BiTFI-TimesFM3 --smoke
.venv_tfm3/bin/python 02_model/hourly_revision.py controls --model BiTFI-TimesFM3-fwd --smoke
for shard in 0 1; do
  .venv_tfm3/bin/python 02_model/hourly_revision.py controls --model BiTFI-TimesFM3-fwd --shard "$shard" --shards 2
done
.venv/bin/python 02_model/hourly_revision.py controls --model BiTFI-Chronos2 --smoke
.venv/bin/python 02_model/hourly_revision.py controls --model BiTFI-Chronos2
```

The three refinement shards report Step 1 alone and 1, 2, 3 and 5 passes; analysis selects the validation-chosen depth. Independent shards can run in parallel on distinct GPUs. The two matched SAITS controls are already evaluated by `saits-evaluate`. They use the same 24-channel capacity, with reference slots unavailable in the local-only control. MOMENT reconstructs channels independently; adaptation tunes only its reconstruction head.

## Auxiliary comparisons, statistics and figures

```bash
.venv_tfm3/bin/python 02_model/hourly_revision.py auxiliary --aux-action additional
.venv_tfm3/bin/python 02_model/hourly_revision.py auxiliary --aux-action matched
.venv_tfm3/bin/python 02_model/hourly_revision.py auxiliary --aux-action univariate --model TimesFM3.0
for model in Chronos2 TimesFM2.5; do
  .venv/bin/python 02_model/hourly_revision.py auxiliary --aux-action univariate --model "$model"
done
for model in BiTFI-TimesFM3 TimesFM3.0-COV-SPA; do
  .venv_tfm3/bin/python 02_model/hourly_revision.py auxiliary --aux-action example --model "$model"
done
for model in SAITS-spatial MOMENT-FT Spatial-Ridge; do
  .venv/bin/python 02_model/hourly_revision.py auxiliary --aux-action example --model "$model"
done
.venv/bin/python 02_model/hourly_analysis.py
.venv/bin/python 02_model/hourly_render.py --results-only
```

The B/C contrast uses identical times and indoor targets. Environmental-change thresholds come from training prefixes; constant-value sensitivity excludes flagged cases without altering the primary results. Fig. 6 chooses the middle coverage-eligible 72-h case after sorting identifiers and timestamps, before consulting predictions. Panel metrics compare BiTFI with the best displayed baseline, including TimesFM3.

Overall scores weight observations within each greenhouse and greenhouses equally. The reporting analysis also provides scenario-specific refinement-depth scores, all five Holm test families, and a sensitivity that averages target variables within each masking case, cases equally within greenhouse, and greenhouses equally. Use `hourly_analysis.py --reporting-only` to regenerate these tables from completed outputs without inference. Bootstrap intervals and paired tests use greenhouses as statistical units. The main display uses the strongest tested SAITS, MOMENT and TimesFM3 input/adaptation settings, with all 21 retained in supplementary outputs. `revision_config.py` defines the common eight-model display. Analysis verifies this choice against the run's actual scores.

## Scope of the release

Source code, environment snapshots and compact aggregate results in `revision_results/` and `revision2_results/` are included. The Smart Farm Korea records are not redistributed (see `../01_data/README.md`), and downloaded pretrained weights, fitted checkpoints, individual prediction arrays, full evaluation case tables and private manuscript files are excluded. Downloaded model caches reside under `03_result/model_cache`; cloning this repository alone does not supply them. Legacy scripts remain for traceability; the hourly entry points above define the current protocol.

Manuscript builders, validation of author-specific TeX and packaging utilities require separately maintained `05_thesis` assets. The included Fig. 2 generator completes an author-drawn draft into an editable vector schematic and needs that draft (`figures/assets/Figure2_author_draft_20260928.pdf`), which is not included. The authored graphical abstract is separate from the computational experiment. Neither schematic provides measured data. The scientific plots are generated from scored predictions and supplied source tables.

## Verification and supplementary sensitivity

`verify_selected_production.py` checks the selected five-pass path against stored predictions, hidden-target perturbations, reversed batches and serial inference. Set matrix precision to `highest` after TimesFM3 model construction. `run_verified_single_call.py` documents a historical isolated control that retains 128-h auxiliary initialization; it explicitly restores rolling mode before overriding target calls. The current full-gap primary evaluation also changes auxiliary initialization and is recorded separately. `verified_sensitivity_analysis.py --help` describes percentile normalization and common-bound clipping of saved outputs. The latter preserves primary predictions. Revision-specific drivers retain interrupted attempts and pre-update artifacts; those attempts are not successful model results. `run_verified_training.py` and `run_verified_autogluon.py` require a copied, completed baseline experiment and document the isolated six-model update. They are not substitutes for the fresh-run commands above. The compact public payload excludes those large pre-update artifacts; see `revision_results/README.md` for the exact rerun and reuse scope.

`hourly_render.py --results-only` preserves dataset and author-drawn framework figures. Manuscript prose is author-edited; `integrate_verified_tables.py` refreshes numerical table bodies without replacing it. Full manuscript template generators are historical build helpers and must not overwrite the current author-edited documents.

## Revision 2

The second revision adds one correction and five analyses. They reuse the stored benchmark predictions of the 21 configurations, and the only new model training is the SAITS seed repeat, which keeps the original seed-42 results. The correction concerns past-only univariate adapters whose pre-gap context holds no target observation. The analyses are BiTFI information-source and reference controls, equal-information regression controls, SAITS training-seed repeats, derived VPD and daily-radiation indicators with leave-one-greenhouse-out effect sizes, and a diagnosis of the largest indicator error. The drivers run on a copy of a completed full-gap run root named by `BITFI_RUN_ROOT` (here `03_result/revision2_20261019`, copied from `03_result/reevaluation_fullgap_20260929` with the fitted models linked read-only). Several drivers also read stored outputs of the original root, which they never modify. The compact outputs and the numbers that changed are described in [`revision2_results/README.md`](revision2_results/README.md).

TimesFM3 steps use `.venv_tfm3` and set `torch.set_float32_matmul_precision('highest')` after model construction, as in the main evaluation. GPU steps use the single device given by `CUDA_VISIBLE_DEVICES`. Paths in the table are relative to the run root unless they start with `01_data/` or `third_party/`.

| Script | Purpose | Main inputs | Main outputs | Environment |
| :--- | :--- | :--- | :--- | :--- |
| `revision2_m4_fix.py` | Applies the no-history rule of `run_clean_evaluation.no_history_fill` (labeled M4) to stored outputs through the actions `audit`, `verify`, `rescore`, `univariate`, `reaggregate` and `compare` | Stored `evaluation/` outputs of TimesFM3.0, TimesFM2.5 and the four AutoGluon models, `univariate_backbone_comparison/` and the original root | `m4_fix/`, rescored `evaluation/<model>/` files with `*_before_m4.*` backups, `analysis/` and `evaluation/_m4_fix_complete.json` | `.venv`, CPU |
| `verified_sensitivity_analysis.py` | Normalization, clipping, quality and reference sensitivity of the corrected root | Run root and `01_data/meta_data.xlsx` | `verification_20260929/sensitivity_m4/` | `.venv`, CPU |
| `revision2_bitfi_reference_audit.py` | Recomputes the reference selection, fixes the five most-used reference greenhouses and records the random draws | Run root and `quality/reference_availability.csv` | `revision/bitfi_variants/reference_audit.csv` and `top5_reference_facilities.json` | `.venv`, CPU |
| `revision2_bitfi_variants.py` with `revision2_reference_bank.py` | Six BiTFI variants that change only the refinement covariates, with a reproduction check (`--repro`), hidden-target smoke checks (`--smoke`), shards and merging (`--merge`) | Run root, the audit outputs and the stored BiTFI predictions of the original root | `revision/bitfi_variants/<variant>/` and `evaluation/<variant>/` | `.venv_tfm3`, GPU; `--merge` with `.venv` on CPU |
| `revision2_bitfi_queue.py` | Runs the variant jobs one at a time on one GPU | Job list of `repro`, `smoke:<variant>` and `shard<i>:<variant>` | `revision/bitfi_variants/logs/` | `.venv` launcher for `.venv_tfm3` jobs, GPU index 1 or 2 |
| `revision2_bitfi_variants_analysis.py` | Summaries and paired tests of the variants | `evaluation/BiTFI-*`, `revision/bitfi_variants/` and `analysis/refinement_cases.csv` | `analysis/revision2/bitfi_variants_*` | `.venv`, CPU |
| `revision2_covreg.py` | Equal-information LightGBM and ridge controls with smoke checks (`--smoke`), a covariate-block identity check (`--check-block`), the full run and a summary (`--analyze`) | Run root, the BiTFI covariate blocks and results of the original root | `evaluation/CovReg-*/`, `smoke/CovReg-*/` and `analysis/revision2/covreg_*` | `.venv`, CPU with at most 20 workers; `--check-block` with `.venv_tfm3` on CPU |
| `revision2_saits_seeds.py` with `revision_saits_spatial.py --seed --out-root --suffix --only` | SAITS training-seed repeats through the actions `run`, `smoke` and `analyze` | Fixed corpus in `revision/saits_information/` and `third_party/SAITS` | `revision/saits_seeds/` and `analysis/revision2/saits_seeds_*` | `.venv`, GPU for `run` and `smoke` |
| `revision2_indicators.py` | VPD and daily-radiation indicators and leave-one-greenhouse-out effect sizes | Stored predictions of nine configurations and of the regression controls, `data/<site>.pkl`, `analysis/all_site_scores.csv` and `evaluation/_m4_fix_complete.json` | `analysis/revision2/indicators_*`, `loo_effect_sizes*` and `indicators_provenance.json` | `.venv`, CPU |
| `revision2_failure_case.py` | Pass-by-pass directional diagnosis of mask 2727 and, with `--screen`, the post-gap run screen of every scored variable-case | Stored predictions of BiTFI, `BiTFI-cross`, `BiTFI-local` and forward-only BiTFI | `analysis/revision2/failure_case_2727/` | `.venv_tfm3`, GPU; `--screen` with `.venv` on CPU |
| `figures/revision2_figures.py` | Figures through the actions `existing`, `new`, `final` and `compare` | Tables of one run root | `figures/`, `figures/Supplementary/`, `figures/source_data/` and `figures/revision2/` | `.venv`, CPU, Liberation Sans and poppler-utils |

```bash
export BITFI_RUN_ROOT="$PWD/03_result/revision2_20261019"
export BITFI_CLEAN=1 JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1
export HF_HOME="$PWD/03_result/model_cache/huggingface"

# No-history correction, re-aggregation and sensitivity
for step in audit verify rescore univariate reaggregate; do
  .venv/bin/python 02_model/revision2_m4_fix.py "$step"
done
(cd 02_model && ../.venv/bin/python verified_sensitivity_analysis.py --root "$BITFI_RUN_ROOT" --output "$BITFI_RUN_ROOT/verification_20260929/sensitivity_m4")
.venv/bin/python 02_model/revision2_m4_fix.py compare --save "$BITFI_RUN_ROOT/m4_fix/changed_values_analysis.csv"

# BiTFI information-source and reference controls
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

# Equal-information regression controls
for m in CovReg-LGBM CovReg-Ridge; do
  .venv/bin/python 02_model/revision2_covreg.py --model "$m" --smoke
done
.venv_tfm3/bin/python 02_model/revision2_covreg.py --check-block
for m in CovReg-LGBM CovReg-Ridge; do
  .venv/bin/python 02_model/revision2_covreg.py --model "$m" --workers 20
done
.venv/bin/python 02_model/revision2_covreg.py --analyze

# SAITS training seeds
CUDA_VISIBLE_DEVICES=0 .venv/bin/python 02_model/revision2_saits_seeds.py run
CUDA_VISIBLE_DEVICES=0 .venv/bin/python 02_model/revision2_saits_seeds.py smoke
.venv/bin/python 02_model/revision2_saits_seeds.py analyze

# Indicators and leave-one-greenhouse-out effect sizes (after the correction)
.venv/bin/python 02_model/revision2_indicators.py

# Diagnosis of mask 2727 and post-gap screen
CUDA_VISIBLE_DEVICES=0 .venv_tfm3/bin/python 02_model/revision2_failure_case.py
.venv/bin/python 02_model/revision2_failure_case.py --screen

# Figures
.venv/bin/python 02_model/figures/revision2_figures.py existing --with-s6 --aligned-labels
.venv/bin/python 02_model/figures/revision2_figures.py final
.venv/bin/python 02_model/figures/revision2_figures.py new --only s13 s14
```

`revision2_m4_fix.py` keeps backups of every replaced file next to it, so repeated runs give the same result. Its `locate` action looks up the affected manuscript strings and needs the private manuscript sources. The original variant runs used `revision2_bitfi_queue.py --gpu <1|2> --jobs ...`, which issues the same calls one at a time and merges each variant after both shards finish. `revision2_indicators.py` refuses to write final outputs before `evaluation/_m4_fix_complete.json` exists. The diagnosis of mask 2727 needs the stored predictions of the `BiTFI-cross` and `BiTFI-local` variants. The `final` figure action also copies an author-drawn reconstruction example from the private manuscript folder, so outside that setting run `final --only fig6 fig7 s11 s12 s15 s16`. The keys after `--only` name figure builders in the script and do not follow the manuscript numbering.

Other code changes in this release are additive or remove unused inputs.

- `preprocessing.py` and the scripts that call it no longer load the two auxiliary sensor columns (wind speed and indoor radiation), which no model used. The five target series are unchanged.
- `data_split.py` resolves and checks the frozen split in `01_data/split.json` instead of drawing a new one.
- `data_usage_periods.py` adds optional per-site recording bounds. `clean_protocol.py` applies them only when the run root holds `data_usage_periods.json` and only to a run root that has not been prepared, and the protocol is unchanged without that file. Run `.venv/bin/python -m unittest discover -s 02_model -p test_data_usage_periods.py` to test the bounds.
- `hourly_analysis.py`, `hourly_auxiliary.py`, `hourly_render.py` and `package_hourly_publication.py` skip the additional-site outputs and figures that a run root's `publication_scope.json` excludes.
- `bitfi_refinement_ablation.py` (model attribute `use_local`) and `revision_saits_spatial.py` (seed and output-folder options) gained options whose defaults keep the earlier behavior, and `figures/prism_style.py` adds labels and colors for the new controls.

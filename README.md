# BiTFI: greenhouse time-series imputation

Source code and compact aggregate results accompanying **Bidirectional imputation of missing greenhouse sensor data using frozen time-series foundation models**.

BiTFI initializes missing values from observations before and after a gap, then performs validation-selected synchronous covariate-assisted refinement using a frozen forecasting backbone. This is retrospective imputation, so post-gap observations may be used.

## Contents

- [`01_data/`](01_data/) holds the frozen candidate site split (`split.json`) and explains how to obtain the Smart Farm Korea records, which this repository does not redistribute.
- [`02_model/`](02_model/) holds preprocessing, BiTFI, comparison models, validation-based context selection, evaluation and plotting sources.
- [`02_model/revision_results/`](02_model/revision_results/) holds compact aggregate results and execution/validation records for the September 2026 benchmark.
- [`02_model/revision2_results/`](02_model/revision2_results/) holds the compact results of the second revision, which correct the September values of six configurations and add new controls and indicators.
- [`02_model/revision3_results/`](02_model/revision3_results/) holds the compact results of the third revision, which tunes the equal-information LightGBM control on validation gaps.

The analysis retains 24 training and 10 test greenhouses. Records are reindexed to a complete hourly grid; absent timestamp rows stay missing and gaps preserve elapsed time. Context selection uses univariate validation gaps from training sites; all three tested backbones select a maximum of 1,900 hourly positions from the candidate range. A separate refinement-depth validation uses 1,896 masks from the final 20% of 21 training sites and selects five passes among 1, 2, 3 and 5. The target site is excluded from its reference bank. The resulting test benchmark uses 3,421 mask jobs (6,205 variable-level cases).

## Running the experiments

See [`02_model/README.md`](02_model/README.md) for entry points, runtime requirements and the order of intermediate artifacts. Pinned environment snapshots and the external SAITS revision are recorded in [`02_model/environments/`](02_model/environments/). Model-specific environments and pretrained checkpoints are needed. Frozen input filenames are listed in [`01_data/split.json`](01_data/split.json). Place the downloaded records under these names in `01_data/` and resolve them against that directory when initializing the original experiment paths.

The revised main comparison contains 21 configurations, with a common 8-setting display set. Additional experiments test initialization/refinement depth, matched SAITS inputs with and without reference-greenhouse channels, and nine further sites with indoor temperature, RH and CO2. The main display uses SAITS (local + cross), MOMENT (head-tuned), and TimesFM3 (local + cross); the complete inventory retains their other input/adaptation settings. Agricultural interpretation concerns the five measured variables.

The current TimesFM2.5 and TimesFM3 comparison adapters request complete gaps, including auxiliary univariate initialization, using the validation-selected 1,900-hour context. BiTFI retains its five-pass reconstruction.

This repository provides source scripts, environment snapshots and compact aggregate results. The greenhouse records, model weights, trained checkpoints, individual prediction arrays and manuscript files are not included. Some scripts consume outputs from earlier preprocessing, training and evaluation stages; the compact summaries do not replace those full intermediate artifacts. Historical code is identified in the model directory documentation.

## Second revision

The second revision corrects the past-only univariate adapters (TimesFM2.5, TimesFM3 and the four AutoGluon models) for gaps whose pre-gap context holds no target observation. Such gaps now keep the masked-input interpolation instead of a forecast from a zero-filled context. The correction changes 10 of the 6,205 variable-cases, all at one test greenhouse, and the stored predictions were rescored without refitting. The six configurations change accordingly (LightGBM, for example, from an NMAE of 0.0611 to 0.0584), while BiTFI and the other 15 configurations keep their values. The revision also adds BiTFI information-source and reference controls, equal-information regression controls, SAITS training-seed repeats, derived VPD and daily-radiation indicators with leave-one-greenhouse-out effect sizes, and a diagnosis of the largest indicator error. [`02_model/revision2_results/README.md`](02_model/revision2_results/README.md) lists the changed values, the files and the commands that produce them, and the Revision 2 section of [`02_model/README.md`](02_model/README.md) describes the scripts.

## Third revision

The third revision, prepared for submission to *Computers and Electronics in Agriculture*, selects the settings of the equal-information LightGBM control on the same validation gaps that selected BiTFI's refinement depth and replaces the fixed-settings control with it. The tuned control reached an NMAE of 0.0361 (fixed settings 0.0372), and BiTFI remained 4.29% lower (95% CI 1.48–7.58%, lower at 8 of 10 greenhouses). BiTFI and the other configurations keep their values. [`02_model/revision3_results/README.md`](02_model/revision3_results/README.md) gives the protocol, the results and the commands.

## Data source

Environmental records were obtained from [Smart Farm Korea](https://www.smartfarmkorea.net/). They are available from the provider under its terms of use and are not redistributed in this repository. See [`01_data/README.md`](01_data/README.md) for the files and the layout that the code expects.

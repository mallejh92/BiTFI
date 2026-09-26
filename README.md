# BiTFI: greenhouse time-series imputation

Data and source code accompanying **Bidirectional imputation of missing greenhouse sensor data using frozen time-series foundation models**.

BiTFI initializes missing values from observations before and after a gap, then performs validation-selected synchronous covariate-assisted refinement using a frozen forecasting backbone. This is retrospective imputation: post-gap observations may be used.

## Contents

- [`01_data/`](01_data/): Smart Farm Korea environmental records, site metadata, exploratory notebook and frozen candidate site split.
- [`02_model/`](02_model/): preprocessing, BiTFI, comparison models, validation-based context selection, evaluation and plotting sources.

The analysis retains 24 training and 10 test greenhouses. Records are reindexed to a complete hourly grid; absent timestamp rows stay missing and gaps preserve elapsed time. Context selection uses univariate validation gaps from training sites; all three tested backbones select a maximum of 1,900 hourly positions from the candidate range. A separate refinement-depth validation uses 1,896 masks from the final 20% of 21 training sites and selects five passes among 1, 2, 3 and 5. The target site is excluded from its reference bank. The resulting test benchmark uses 3,421 mask jobs (6,205 variable-level cases).

## Running the experiments

See [`02_model/README.md`](02_model/README.md) for entry points, runtime requirements and the order of intermediate artifacts. Pinned environment snapshots and the external SAITS revision are recorded in [`02_model/environments/`](02_model/environments/). Model-specific environments and pretrained checkpoints are needed. Frozen input filenames are listed in [`01_data/split.json`](01_data/split.json); resolve them against the local data directory when initializing the original experiment paths.

The revised main comparison contains 21 configurations, with a common 8-setting display set. Additional experiments test initialization/refinement depth, matched SAITS inputs with and without reference-greenhouse channels, and nine further sites with indoor temperature, RH and CO2. The main display uses SAITS (local + cross), MOMENT (head-tuned), and TimesFM3 (local + cross); the complete inventory retains their other input/adaptation settings. Agricultural interpretation concerns the five measured variables.

This repository provides data and source scripts. It does not include model weights, trained checkpoints, generated result archives or manuscript files. Some scripts consume outputs from prior preprocessing/training/evaluation stages, so the repository is not a precomputed result bundle. Historical code is identified in the model directory documentation.

## Data source

Environmental records were obtained from [Smart Farm Korea](https://www.smartfarmkorea.net/). Original provider terms and attribution continue to apply to the distributed data. See [`01_data/README.md`](01_data/README.md).

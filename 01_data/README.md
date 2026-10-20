# Greenhouse environmental records

The greenhouse records analyzed in this study come from [Smart Farm Korea](https://www.smartfarmkorea.net/). This repository does not redistribute them. Obtain the hourly environmental records from Smart Farm Korea under the provider's [terms of use](https://www.smartfarmkorea.net/common/policyPop.do?policyTy=000001), which govern access to and reuse of every download.

## Expected layout

`split.json` lists the frozen site split by filename, and each filename is a Smart Farm Korea site identifier such as `PF_0000574_01.xlsx`. Save the record of each listed site in this directory under that filename. Subdirectories are searched as well, but each name must match exactly one file. From the repository root, `.venv/bin/python 02_model/initialize_public_paths.py --check-only` reports whether every listed file can be resolved without writing anything.

`split.json` names 10 test sites and 33 further sites. The 24 further sites that record all five target variables form the training pool, and the `meta` entry of `split.json` records the selection rule. `02_model/data_split.py` resolves and checks this frozen split instead of drawing a new one, because a new draw would change every reported result.

## File contents

Keep each spreadsheet as downloaded, with its original column labels. The code finds the timestamp column (`수집일`) and selects the five target variables by their original labels (`내부-내부온도`, `외부-외부온도`, `내부-내부습도`, `내부-내부CO2` and `외부-외부일사량`). Columns that were renamed or translated are not recognized, and all other columns are ignored.

The code also handles the recording period. It reindexes each record to a complete hourly grid, keeps absent timestamps as missing values, applies physical-range screening and derives the chronological training prefix and the evaluation masks. Trimming, resampling or filling the files beforehand would change the inputs. Physical screening and raw-observation masking are implemented in `02_model/clean_protocol.py` and `02_model/preprocessing.py`. Original missing values are never used as scored truth.

Two reporting scripts, `02_model/verified_sensitivity_analysis.py` (crop matches of the selected references) and `02_model/build_selected_context_assets.py`, also read a site metadata table, `meta_data.xlsx`, in this directory. Its columns are `name`, `plant`, `Covering`, `type`, `height (m)`, `width (m)`, `length (m)`, `area (m2)`, `Start` and `End`. The table is not included, and the benchmark itself does not read it.

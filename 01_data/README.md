# Greenhouse environmental records

Source: Smart Farm Korea, https://www.smartfarmkorea.net/ . This directory contains the downloaded greenhouse spreadsheets and site metadata used by the project; redistribution does not change the original provider’s terms or attribution.

`PF_*.xlsx` are hourly site records; `meta_data.xlsx` contains site identifiers, crops, structural characteristics and recording periods. Not every downloaded site is eligible for every analysis. The manuscript retains 24 training and 10 test greenhouses after variable-availability screening. `split.json` lists the frozen candidate split by filename; resolve names against this directory. `EDA.ipynb` is the exploratory notebook.

The sheets retain their original column labels. Physical screening and raw-observation masking are implemented in `02_model/clean_protocol.py` and preprocessing helpers. Original missing values must not be used as scored truth.

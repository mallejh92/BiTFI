"""Reproduce a predeclared illustrative BiTFI case (first test site, C/72h/r0)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from preprocessing import preprocess_file, TARGET_COLS
from masking_v2 import create_gap_masks, SCENARIO_CONFIGS, RANDOM_SEED
from run_comparison import _stable_seed
from spatial import NeighborBank
from dafi import DAFITimesFM3
ROOT=Path(__file__).resolve().parents[1]
split=json.loads((ROOT/"03_result/comparison/split.json").read_text())
fp=split["test"][0];res=preprocess_file(fp,include_covariates=True)
full=res["data"];cols=list(TARGET_COLS.values());valid=full.notna().astype("float32")
name=res["name"];scenario="C";gap=72
mr=create_gap_masks(full,valid,scenario,gap,cols,context_len_h=1440,n_repeats=10,
    random_seed=RANDOM_SEED+_stable_seed(f"{name}{scenario}{gap}"))[0]
bank=NeighborBank([Path(p) for p in split["train"]],cols)
model=DAFITimesFM3(bank=bank,context_len=1440,name="DAFI-TimesFM3")
if not model._ready():raise RuntimeError("BiTFI backend not ready")
model.set_greenhouse(name)
base=model.compute_base(full[cols],valid[cols])
art=mr.artificial_mask[cols].eq(0)
pred=model.impute_artificial(mr.masked_data[cols],mr.effective_mask[cols],art,base)
lo=max(0,mr.gap_start_idx-72);hi=min(len(full),mr.gap_end_idx+73)
rows=[]
for c in cols:
    rows.append(pd.DataFrame({"datetime":full.index[lo:hi],"variable":c,
        "truth":mr.ground_truth[c].iloc[lo:hi].to_numpy(),
        "prediction":pred[c].iloc[lo:hi].to_numpy(),"artificial":art[c].iloc[lo:hi].to_numpy()}))
out=ROOT/"03_result/comparison_dafi_tfm3/examples";out.mkdir(exist_ok=True)
pd.concat(rows).to_csv(out/f"{name}_C_Tin-Tout-RH-CO2-Rad_72h.csv",index=False)
(out/"selection.json").write_text(json.dumps({"selection":"first test site, scenario C, 72h, repeat 0; selected before inspecting model errors",
    "greenhouse":name,"start":str(full.index[mr.gap_start_idx]),"end":str(full.index[mr.gap_end_idx])},indent=2))
print("Saved BiTFI example",name,flush=True)

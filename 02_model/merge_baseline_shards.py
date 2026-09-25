"""Merge independent greenhouse shards only after every shard completes."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
KEY=["greenhouse","model","scenario","masked_vars","gap_length_h","repeat","group_type","group_value","variable"]
p=argparse.ArgumentParser();p.add_argument("--model",default="AG-LightGBM");p.add_argument("--shards",type=int,default=5)
a=p.parse_args();out=ROOT/"03_result"/("comparison_"+a.model.lower())
folders=[out/"shards"/str(i) for i in range(a.shards)]
for f in folders:
    if not (f/"complete.json").exists():raise RuntimeError(f"Incomplete shard: {f}")
d=pd.concat([pd.read_csv(f/"results.csv") for f in folders],ignore_index=True)
if d.duplicated(KEY).any():raise ValueError("Duplicate evaluation cells")
ref=pd.read_csv(ROOT/"03_result/comparison_saits/results.csv")
keys=[k for k in KEY if k!="model"]
if set(map(tuple,d[keys].to_numpy()))!=set(map(tuple,ref[keys].to_numpy())):
    raise ValueError("Evaluation cells differ from SAITS")
m=pd.concat([pd.read_csv(f/"mask_manifest.csv") for f in folders],ignore_index=True)
refm=pd.read_csv(ROOT/"03_result/comparison_saits/mask_manifest.csv")
if set(map(tuple,m.to_numpy()))!=set(map(tuple,refm.to_numpy())):
    raise ValueError("Actual gap locations differ")
if (out/"results.csv").exists() and not (out/"results_unsharded_partial.csv").exists():
    shutil.copy2(out/"results.csv",out/"results_unsharded_partial.csv")
d.to_csv(out/"results.csv",index=False);m.to_csv(out/"mask_manifest.csv",index=False)
ed=out/"examples";ed.mkdir(exist_ok=True)
for f in folders:
    for x in (f/"examples").glob("*.csv"):shutil.copy2(x,ed/x.name)
q=d[d.group_type=="all"].groupby("greenhouse").apply(lambda z:np.average(z.NMAE,weights=z.n_eval),include_groups=False)
info={"model":a.model,"n_greenhouses":len(q),"rows":len(d),"all_cells":int((d.group_type=="all").sum()),
      "NMAE":q.mean(),"SD":q.std(),"mask_manifest_matches_SAITS":True}
(out/"complete.json").write_text(json.dumps(info,indent=2))
print(json.dumps(info,indent=2))

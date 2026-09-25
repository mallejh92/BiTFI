"""Rebuild the equal-input S6 control at the new common 1900-h context."""
from pathlib import Path
import argparse,json,pickle,hashlib
import numpy as np,pandas as pd
import clean_protocol as cp
import run_controlled_context_sweep as sweep
OUT=cp.ROOT/'03_result/reevaluation_context_20260925/univariate_backbone_comparison'
def prepare():
 if OUT.is_symlink():OUT.unlink()
 OUT.mkdir(exist_ok=True)
 meta=pd.read_csv(cp.OUT/'univariate_backbone_comparison/cases.csv');meta['context_len']=1900;meta['case_id']=np.arange(len(meta));records=[];sites={};checks=0
 for r in meta.itertuples():
  if r.greenhouse not in sites:sites[r.greenhouse]=cp.site(r.greenhouse)
  o=sites[r.greenhouse];gs=o['data'].index.get_loc(pd.Timestamp(r.start_time));ge=gs+r.gap_length_h;v=r.variable
  x=o['data'][v].iloc[max(0,gs-1900):gs].interpolate(limit_direction='both').ffill().bfill().to_numpy(np.float32)
  y=o['data'][v].iloc[gs:ge].to_numpy(np.float32);assert np.isfinite(x).all() and np.isfinite(y).all()
  records.append((x,np.empty((0,len(x)+len(y)),np.float32),y,r.nmae_factor));checks+=1
 with (OUT/'inputs.pkl').open('wb') as f:pickle.dump({1900:records},f)
 meta.to_csv(OUT/'cases.csv',index=False)
 (OUT/'protocol.json').write_text(json.dumps(dict(protocol='validation-context-20260925',context_hours=1900,cases=len(meta),same_case_manifest=True,information='Univariate past-only; full gap predicted in one call; shorter available history used identically across models',inputs_sha256=hashlib.sha256((OUT/'inputs.pkl').read_bytes()).hexdigest()),indent=2))
def run(m):
 sweep.OUT=OUT;sweep.CONTEXTS=[1900];sweep.run(m,arms=['univariate'],gaps=[6,12,24,72,168])
 d=pd.read_csv(OUT/f'{m}_univariate_ctx1900.csv');d['model']=m+'-UNI';d.to_csv(OUT/f'{m}_results.csv',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--model');a=p.parse_args()
 if a.prepare:prepare()
 else:run(a.model)

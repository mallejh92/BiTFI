"""Data-only checks and quality flags fixed before residual analysis."""
import json,hashlib
import numpy as np,pandas as pd
import clean_protocol as cp
from preprocessing import assert_hourly

def main():
 out=cp.OUT/'quality';out.mkdir(exist_ok=True);manifest=pd.read_csv(cp.OUT/'mask_manifest.csv');counts=[];runs=[];thresholds={v:[] for v in cp.COLS};site_objects={}
 for r in pd.read_csv(cp.OUT/'sites.csv').itertuples():
  o=cp.site(r.name);raw=o['data_raw'][cp.COLS];assert_hourly(raw.index);site_objects[r.name]=o
  flags=np.zeros(raw.shape,bool)
  for j,v in enumerate(cp.COLS):
   x=raw[v].to_numpy();bounds=np.r_[0,np.flatnonzero(~(np.isfinite(x[1:])&np.isfinite(x[:-1])&(x[1:]==x[:-1])))+1,len(x)]
   for a,b in zip(bounds[:-1],bounds[1:]):
    if b-a>=48 and np.isfinite(x[a]):
     flags[a:b,j]=True;runs.append(dict(greenhouse=r.name,group=r.group,variable=v,start=str(raw.index[a]),end=str(raw.index[b-1]),hours=b-a,value=x[a]))
   if r.group=='train':thresholds[v].extend(raw[v].iloc[:o['cut']].diff().abs().dropna().tolist())
   counts.append(dict(greenhouse=r.name,group=r.group,variable=v,hourly_positions=len(x),observations=int(np.isfinite(x).sum()),missing_fraction=float(np.isnan(x).mean()),constant_flag_hours=int(flags[:,j].sum())))
  np.savez_compressed(out/f'{r.name}_flags.npz',constant=flags)
 for r in manifest.itertuples():
  o=site_objects[r.greenhouse];idx=o['data'].index
  assert idx[r.end_idx]-idx[r.start_idx]==pd.Timedelta(hours=r.gap_length_h-1)
  assert r.end_idx-r.start_idx+1==r.gap_length_h
  assert o['data_raw'][r.masked_vars.split(',')].iloc[r.start_idx:r.end_idx+1].notna().all().all()
  for start,end in [(max(0,r.start_idx-1900),r.start_idx),(r.end_idx+1,min(len(idx),r.end_idx+1901))]:assert_hourly(idx[start:end])
 pd.DataFrame(counts).to_csv(out/'hourly_missingness.csv',index=False);pd.DataFrame(runs).to_csv(out/'constant_runs.csv',index=False)
 th={v:float(np.quantile(x,.75)) for v,x in thresholds.items()};(out/'change_thresholds.json').write_text(json.dumps(th,indent=2))
 p=dict(sites=len(site_objects),mask_cases=len(manifest),variable_cases=sum(len(x.split(',')) for x in manifest.masked_vars),hourly_index_checks=True,gap_duration_checks=True,context_duration_checks=True,duplicate_or_off_hour_policy='Reject; never average or silently discard',constant_flag='At least 48 consecutive hourly positions with the same finite value; interrupted by every missing hour. Flags are sensitivity indicators, not confirmed faults; primary data retained.',change_strata='Absolute one-hour change above versus at/below training-prefix variable-specific 75th percentile; adjacent genuinely observed hours only',radiation_strata='Zero versus positive physically screened irradiance; not a claim of astronomical day/night',thresholds=th,manifest_sha256=hashlib.sha256((cp.OUT/'mask_manifest.csv').read_bytes()).hexdigest())
 (out/'protocol.json').write_text(json.dumps(p,indent=2));print(json.dumps(p,indent=2))
if __name__=='__main__':main()

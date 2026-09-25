"""Validation-only univariate backbone context selection, fixed before inference."""
import argparse,json,pickle,hashlib,zlib
from pathlib import Path
import numpy as np,pandas as pd
import clean_protocol as cp
import run_controlled_context_sweep as sweep
OUT=cp.ROOT/'03_result/context_validation_20260925'
CONTEXTS=[96,168,336,720,1080,1440,1900]
GAPS=[6,12,24,72,168]
def prepare():
 OUT.mkdir(exist_ok=True);rows=[];objects={}
 for site in pd.read_csv(cp.OUT/'sites.csv').query("group=='train'").itertuples():
  obj=cp.site(site.name);objects[site.name]=obj;cut=obj['cut']
  for v in cp.COLS:
   raw=obj['data_raw'][v];span=float(raw.iloc[:cut].max()-raw.iloc[:cut].min())
   if not np.isfinite(span) or span<=0:continue
   for h in GAPS:
    valid=raw.notna().to_numpy();count=np.convolve(valid.astype(int),np.ones(h,dtype=int),'valid')
    candidates=np.flatnonzero((count==h)&(np.arange(len(count))>=max(cut,max(CONTEXTS))))
    seed=42+zlib.crc32(f'{site.name}|{v}|{h}|validation'.encode())%100000
    rng=np.random.RandomState(seed);used=set();repeat=0
    for gs in rng.permutation(candidates):
     if any(i in used for i in range(gs,gs+h)):continue
     # Every candidate context needs at least one genuine target observation.
     if not valid[gs-min(CONTEXTS):gs].any():continue
     used.update(range(gs,gs+h));rows.append(dict(case_id=len(rows),greenhouse=site.name,scenario='A',masked_vars=v,gap_length_h=h,repeat=repeat,start_idx=int(gs),end_idx=int(gs+h-1),start_time=str(raw.index[gs]),validation_start=int(cut),training_range=span));repeat+=1
     if repeat==3:break
 meta=pd.DataFrame(rows);assert len(meta)>0
 records={};checks=0
 for c in CONTEXTS:
  data=[]
  for r in meta.itertuples():
   obj=objects[r.greenhouse];gs,ge=r.start_idx,r.end_idx;v=r.masked_vars
   x=obj['data'][v].iloc[gs-c:gs].interpolate(limit_direction='both').ffill().bfill().to_numpy(np.float32)
   truth=obj['data'][v].iloc[gs:ge+1].to_numpy(np.float32)
   assert gs>=obj['cut'] and len(x)==c and np.isfinite(x).all() and np.isfinite(truth).all()
   poison=obj['data'][v].copy();poison.iloc[gs:]=99999
   xx=poison.iloc[gs-c:gs].interpolate(limit_direction='both').ffill().bfill().to_numpy(np.float32)
   np.testing.assert_array_equal(x,xx);checks+=1
   data.append((x,np.empty((0,c+len(truth)),np.float32),truth,float(obj['scaler'][v].data_range_[0])/r.training_range))
  records[c]=data
 meta.to_csv(OUT/'cases.csv',index=False)
 with (OUT/'inputs.pkl').open('wb') as f:pickle.dump(records,f)
 protocol=dict(version='validation-context-20260925',target_split='Last chronological 20% of training greenhouses only; no test greenhouse',context='Past-only, may include the earlier training prefix and observed earlier validation values; full requested context available for all candidates',contexts=CONTEXTS,gaps=GAPS,variables=cp.COLS,repeats=3,cases=len(meta),sites=sorted(meta.greenhouse.unique()),selection='Minimize univariate NMAE: hour-weighted within greenhouse, then unweighted mean across greenhouses. Exact ties choose shorter context.',normalization='Frozen existing training-prefix global scalers; scoring divided by per-site variable training-prefix raw range',seed=42,feature_poison_checks=checks,arms=['univariate'],inputs_sha256=hashlib.sha256((OUT/'inputs.pkl').read_bytes()).hexdigest())
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol,indent=2))
def select():
 frames=[]
 for m in ['Chronos2','TimesFM2.5','TimesFM3.0']:
  for c in CONTEXTS:
   d=pd.read_csv(OUT/f'{m}_univariate_ctx{c}.csv');frames.append(d)
 d=pd.concat(frames);g=d.assign(w=d.NMAE*d.n_eval).groupby(['model','context','greenhouse'])[['w','n_eval']].sum();g['NMAE']=g.w/g.n_eval
 g.reset_index().to_csv(OUT/'greenhouse_scores.csv',index=False)
 s=g.reset_index().groupby(['model','context']).NMAE.agg(['mean','std','count']).reset_index();s.to_csv(OUT/'summary.csv',index=False)
 selection={m:int(q.sort_values(['mean','context']).iloc[0].context) for m,q in s.groupby('model')}
 (OUT/'selected_contexts.json').write_text(json.dumps(selection,indent=2));print(selection)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--select',action='store_true');p.add_argument('--model');a=p.parse_args()
 if a.prepare:prepare()
 elif a.select:select()
 else:
  sweep.OUT=OUT;sweep.CONTEXTS=CONTEXTS;sweep.run(a.model,arms=['univariate'],gaps=GAPS)

"""Synchronous refinement stages for validation selection and controlled evaluation."""
from collections import defaultdict
import time
import numpy as np,pandas as pd,torch
from bitfi import MIN_SIDE_CTX
from clean_protocol import COLS,masked_case

def infer_stages(model,examples,max_refinements=5,cols=None):
 cols=COLS if cols is None else cols
 states=[];records=[];timings={}
 for obj,row in examples:
  mr=masked_case(obj,row);masked=mr.masked_data[cols];observed=mr.effective_mask[cols].ne(0).values&np.isfinite(masked.values);art=mr.artificial_mask[cols].eq(0)
  est=masked.interpolate(limit_direction='both').ffill().bfill().values.astype(np.float32);obs=masked.values.astype(np.float32);est[observed]=obs[observed];jobs=[]
  for ci,c in enumerate(cols):
   for gs,ge in model._find_gaps(~observed[:,ci]):
    if art[c].iloc[gs:ge+1].any():jobs.append((ci,gs,ge))
  for ci,gs,ge in jobs:est[gs:ge+1,ci]=np.nan
  state=dict(mr=mr,index=masked.index,est=est,obs=obs,observed=observed,jobs=jobs,directional={},neighbors={});states.append(state)
  for ci,gs,ge in jobs:
   h=ge-gs+1;left,right=model._sides(est[:,ci],gs,ge);directions=[]
   if h<=model._max_h():
    if len(left)>=MIN_SIDE_CTX:directions.append(('f',left))
    if model.bidirectional and len(right)>=MIN_SIDE_CTX:directions.append(('b',right[::-1].copy()))
    if not directions:
     if len(left):directions.append(('f',left))
     elif len(right):directions.append(('b',right[::-1].copy()))
   for direction,ctx in directions:records.append((state,(ci,gs,ge),direction,ctx,None,h))
 def forecast(records):
  groups=defaultdict(list)
  for rec in records:groups[(rec[5],0 if rec[4] is None else rec[4].shape[0])].append(rec)
  for (h,ncov),items in groups.items():
   for start in range(0,len(items),16):
    chunk=items[start:start+16];kw={} if not ncov else dict(past_future_covariates=[r[4] for r in chunk],padding_mode='edge')
    if getattr(model,'tfm',None) is not None:
     with torch.inference_mode():out=list(model.tfm.predict_batch(contexts=[r[3] for r in chunk],horizon=h,**kw))
    else:
     from types import SimpleNamespace
     out=[]
     for r in chunk:
      if r[4] is None:y=model._fc_uni(r[3],h)
      else:
       names=[f'v{j}' for j in range(len(cols)) if j!=r[1][0]]
       if model.use_time_covariates:names+=['t_hs','t_hc','t_ds','t_dc']
       names += [f'nb{i}' for i in range(r[4].shape[0]-len(names))]
       y=model._fc_cov(r[3],names,r[4],h)
      out.append(SimpleNamespace(forecast=y))
    for rec,p in zip(chunk,out):
     s,key,direction,*_=rec;y=np.asarray(p.forecast,np.float32).flatten()[:h]
     assert np.isfinite(y).all()
     s['directional'].setdefault(key,{})[direction]=y if direction=='f' else y[::-1].copy()
 def materialize():
  result=[]
  for s in states:
   pred=pd.DataFrame(s['est'].copy(),index=s['index'],columns=cols).interpolate(limit_direction='both').ffill().bfill()
   result.append((s['mr'],pred))
  return result
 torch.cuda.synchronize();start=time.perf_counter();forecast(records)
 for s in states:
  for ci,gs,ge in s['jobs']:
   d=s['directional'].get((ci,gs,ge),{});p=model._fuse(d.get('f'),d.get('b'))
   if p is None:p=pd.Series(s['est'][:,ci]).interpolate(limit_direction='both').ffill().bfill().values[gs:ge+1]
   s['est'][gs:ge+1,ci]=p
 torch.cuda.synchronize();timings[0]=time.perf_counter()-start
 yield 0,materialize(),timings[0]
 for step in range(1,max_refinements+1):
  torch.cuda.synchronize();start=time.perf_counter();records=[]
  for s in states:
   s['directional']={};est=s['est']
   for ci,gs,ge in s['jobs']:
    h=ge-gs+1
    if h>model._max_h():continue
    key=(ci,gs,ge)
    if key not in s['neighbors']:
     nb=None
     if model.use_spatial:nb,_=model.bank.select(cols[ci],s['index'],s['obs'][:,ci],s['observed'][:,ci],k=model.k)
     s['neighbors'][key]=nb
    nb=s['neighbors'][key];others=[j for j in range(len(cols)) if j!=ci];left,right=model._sides(est[:,ci],gs,ge)
    if len(left)>=MIN_SIDE_CTX:
     _,pf=model._cov_block(est,others,nb,s['index'],gs-len(left),ge+1);records.append((s,key,'f',left,pf,h))
    if model.bidirectional and len(right)>=MIN_SIDE_CTX:
     _,pf=model._cov_block(est,others,nb,s['index'],gs,ge+1+len(right));records.append((s,key,'b',right[::-1].copy(),pf[:,::-1].copy(),h))
  forecast(records)
  for s in states:
   # Simultaneous update: every target saw the same previous-pass snapshot.
   out=s['est'].copy()
   for (ci,gs,ge),d in s['directional'].items():out[gs:ge+1,ci]=model._fuse(d.get('f'),d.get('b'))
   s['est']=out
  torch.cuda.synchronize();timings[step]=time.perf_counter()-start
  yield step,materialize(),timings[step]

"""Independent-example batching of the unchanged BiTFI R0/R1 algorithm."""
from collections import defaultdict
import numpy as np,pandas as pd,torch
from bitfi import MIN_SIDE_CTX
from clean_protocol import COLS,masked_case

def infer_batch(model,examples):
 states=[];r0=[]
 for obj,row in examples:
  mr=masked_case(obj,row);masked=mr.masked_data[COLS];mask=mr.effective_mask[COLS];art=mr.artificial_mask[COLS].eq(0)
  obs=masked.values.astype(np.float32);observed=mask.ne(0).values&np.isfinite(obs)
  est=masked.interpolate(limit_direction='both').ffill().bfill().values.astype(np.float32);est[observed]=obs[observed]
  jobs=[]
  for ci,c in enumerate(COLS):
   for gs,ge in model._find_gaps(~observed[:,ci]):
    if art[c].iloc[gs:ge+1].any():jobs.append((ci,gs,ge))
  for ci,gs,ge in jobs:est[gs:ge+1,ci]=np.nan
  state=dict(obj=obj,row=row,mr=mr,masked=masked,index=masked.index,obs=obs,observed=observed,est=est,jobs=jobs,r0={},r1={});states.append(state)
  for ci,gs,ge in jobs:
   h=ge-gs+1;left,right=model._sides(est[:,ci],gs,ge);directions=[]
   if h<=model._max_h():
    if len(left)>=MIN_SIDE_CTX:directions.append(('f',left))
    if model.bidirectional and len(right)>=MIN_SIDE_CTX:directions.append(('b',right[::-1].copy()))
    if not directions:
     if len(left):directions.append(('f',left))
     elif len(right):directions.append(('b',right[::-1].copy()))
   for direction,ctx in directions:r0.append((state,(ci,gs,ge),direction,ctx,None,h))
 def forecast(records,stage):
  groups=defaultdict(list)
  for rec in records:groups[(rec[5],0 if rec[4] is None else rec[4].shape[0],len(rec[3]))].append(rec)
  for (h,ncov,context_length),items in groups.items():
   for start in range(0,len(items),16):
    chunk=items[start:start+16];kwargs={}
    if ncov:kwargs=dict(past_future_covariates=[r[4] for r in chunk],padding_mode='edge')
    with torch.inference_mode():outputs=list(model.tfm.predict_batch(contexts=[r[3] for r in chunk],horizon=h,**kwargs))
    for rec,out in zip(chunk,outputs):
     s,key,direction,*_=rec;p=np.asarray(out.forecast,np.float32).flatten()[:h]
     if direction=='b':p=p[::-1].copy()
     s[stage].setdefault(key,{})[direction]=p
 forecast(r0,'r0');r1=[]
 for s in states:
  est=s['est']
  for ci,gs,ge in s['jobs']:
   preds=s['r0'].get((ci,gs,ge),{});p=model._fuse(preds.get('f'),preds.get('b'))
   if p is None:p=pd.Series(est[:,ci]).interpolate(limit_direction='both').ffill().bfill().values[gs:ge+1]
   est[gs:ge+1,ci]=p
  s['out']=est.copy()
  for ci,gs,ge in s['jobs']:
   h=ge-gs+1
   if h>model._max_h():continue
   nb=None
   if model.use_spatial:nb,_=model.bank.select(COLS[ci],s['index'],s['obs'][:,ci],s['observed'][:,ci],k=model.k)
   others=[j for j in range(len(COLS)) if j!=ci];left,right=model._sides(est[:,ci],gs,ge)
   if len(left)>=MIN_SIDE_CTX:
    _,pf=model._cov_block(est,others,nb,s['index'],gs-len(left),ge+1);r1.append((s,(ci,gs,ge),'f',left,pf,h))
   if model.bidirectional and len(right)>=MIN_SIDE_CTX:
    _,pf=model._cov_block(est,others,nb,s['index'],gs,ge+1+len(right));r1.append((s,(ci,gs,ge),'b',right[::-1].copy(),pf[:,::-1].copy(),h))
 forecast(r1,'r1');results=[]
 for s in states:
  for (ci,gs,ge),preds in s['r1'].items():s['out'][gs:ge+1,ci]=model._fuse(preds.get('f'),preds.get('b'))
  pred=pd.DataFrame(s['out'],index=s['index'],columns=COLS).interpolate(limit_direction='both').ffill().bfill()
  results.append((s['mr'],pred))
 return results

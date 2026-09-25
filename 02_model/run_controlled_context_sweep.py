"""S2: identical masked inputs across three backbones and two information arms."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse,json,pickle,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import clean_protocol as cp
OUT=cp.OUT/'controlled_context_sweep_20260922'
CONTEXTS=[96,168,336,720,1080,1440,1900]

def inputs(obj,r,c):
 gs,ge=int(r.start_idx),int(r.end_idx);v=r.masked_vars;lo=max(0,gs-c)
 # Only the pre-gap target slice is ever accessed for inference.
 x=obj['data'][v].iloc[lo:gs].interpolate(limit_direction='both').ffill().bfill().to_numpy(np.float32)
 others=[a for a in cp.COLS if a!=v]
 # Scenario A: these four channels are unmasked, including during the gap.
 cov=obj['data'][others].iloc[lo:ge+1].interpolate(limit_direction='both').ffill().bfill().fillna(0).to_numpy(np.float32).T
 dt=obj['data'].index[lo:ge+1];hr=dt.hour.to_numpy();day=dt.dayofyear.to_numpy()
 cal=np.stack([np.sin(2*np.pi*hr/24),np.cos(2*np.pi*hr/24),np.sin(2*np.pi*day/365.25),np.cos(2*np.pi*day/365.25)]).astype(np.float32)
 cov=np.concatenate([cov,cal]);assert np.isfinite(x).all() and np.isfinite(cov).all()
 return x,cov

def prepare():
 OUT.mkdir(exist_ok=True)
 m=pd.read_csv(cp.OUT/'mask_manifest.csv');m=m[(m.scenario=='A')&m.gap_length_h.isin([24,168])&(m.repeat<3)].copy()
 sites={n:cp.site(n) for n in m.greenhouse.unique()};records={};meta=[];audits=0
 for c in CONTEXTS:
  rows=[]
  for r in m.itertuples():
   obj=sites[r.greenhouse];x,cov=inputs(obj,r,c);v=r.masked_vars
   truth=obj['data'][v].iloc[int(r.start_idx):int(r.end_idx)+1].to_numpy(np.float32)
   span=float(obj['data_raw'][v].max()-obj['data_raw'][v].min());scale=float(obj['scaler'][v].data_range_[0])
   rows.append((x,cov,truth,scale/span))
   # Poison hidden target AND all post-gap target values; neither is an input.
   poisoned=dict(obj);poisoned['data']=obj['data'].copy();poisoned['data'].loc[poisoned['data'].index[int(r.start_idx):],v]=99999
   xx,cc=inputs(poisoned,r,c);np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(cov,cc);audits+=1
  records[c]=rows
  print('prepared',c,len(rows),flush=True)
 with (OUT/'inputs.pkl').open('wb') as f:pickle.dump(records,f)
 m.to_csv(OUT/'cases.csv',index=False)
 protocol=dict(contexts=CONTEXTS,cases=len(m),scenarios=['A'],gaps=[24,168],repeats=[0,1,2],arms=['univariate','local_covariates'],covariates='Four other local variables plus hour/day-of-year sin/cos; identical context+gap values for all backbones',target='Pre-gap only; same slice-specific filling, no hidden or post-gap targets',natural_missing='Linear interpolation then endpoint fill within the supplied slice; entirely missing covariate channel becomes zero',context='Last min(requested context, available preceding positions); same across backbones',normalization='Fixed clean-20260911 training-prefix scalers',cross_greenhouse=False,tfm25='Official XReg + TimesFM, ridge=0, normalized per series, each case fitted independently (never pooled across batches)',seed=42,feature_poison_checks=audits,inputs_sha256=hashlib.sha256((OUT/'inputs.pkl').read_bytes()).hexdigest())
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))

def run(name,smoke=False,arms=None,gaps=None):
 arms=arms or ['univariate','local_covariates'];gaps=gaps or [24,168]
 import torch
 torch.set_num_threads(4);torch.manual_seed(42);torch.set_float32_matmul_precision('highest' if name=='TimesFM3.0' else 'high')
 with (OUT/'inputs.pkl').open('rb') as f:records=pickle.load(f)
 meta=pd.read_csv(OUT/'cases.csv');start=time.time()
 if name=='Chronos2':
  from chronos import Chronos2Pipeline
  model=Chronos2Pipeline.from_pretrained('amazon/chronos-2',device_map='cuda',dtype=torch.float32)
 elif name=='TimesFM2.5':
  import timesfm
  model=timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
 else:
  import timesfm
  model=timesfm.TimesFM3Forecaster.from_pretrained('google/timesfm-3.0-pytorch',device='cuda')
 def predict(batch,h,c,arm):
  xs=[a[0] for a in batch]
  if name=='Chronos2':
   if arm=='local_covariates':
    xs=[dict(target=x,past_covariates={f'c{i}':v[:len(x)] for i,v in enumerate(cov)},future_covariates={f'c{i}':v[len(x):] for i,v in enumerate(cov)}) for x,cov,*_ in batch]
   _,means=model.predict_quantiles(inputs=xs,prediction_length=h,context_length=c,cross_learning=False,batch_size=128,quantile_levels=[.1,.5,.9])
   return np.stack([p.detach().float().cpu().numpy().reshape(-1)[:h] for p in means])
  if name=='TimesFM2.5':
   if arm=='univariate':return np.asarray(model.forecast(horizon=h,inputs=xs)[0])[:,-h:]
   outputs=[]
   # Official XReg pools a supplied batch; one-case calls prevent cross-case fitting.
   for x,cov,*_ in batch:
    p,_=model.forecast_with_covariates(inputs=[x],dynamic_numerical_covariates={f'c{i}':[v] for i,v in enumerate(cov)},xreg_mode='xreg + timesfm',ridge=0.,force_on_cpu=True)
    outputs.append(np.asarray(p[0])[:h])
   return np.stack(outputs)
  # Mixed-length padding changes native covariate forecasts; group exact lengths.
  result=[None]*len(batch)
  for length in sorted(set(map(len,xs))):
   ii=[i for i,x in enumerate(xs) if len(x)==length]
   kwargs={} if arm=='univariate' else dict(past_future_covariates=[batch[i][1] for i in ii],padding_mode='edge')
   pp=model.predict_batch(contexts=[xs[i] for i in ii],horizon=h,**kwargs)
   for i,prediction in zip(ii,pp):result[i]=np.asarray(prediction.forecast).reshape(-1)[:h]
  return np.stack(result)
 for c in CONTEXTS:
  if name=='TimesFM2.5':
   model.compile(timesfm.ForecastConfig(max_context=c,max_horizon=256,per_core_batch_size=16,normalize_inputs=True,use_continuous_quantile_head=True,force_flip_invariance=True,infer_is_positive=False,fix_quantile_crossing=True,return_backcast=True))
  for arm in arms:
   path=OUT/f'{name}_{arm}_ctx{c}.csv'
   if path.exists() and not smoke:continue
   data=records[c];pred=np.full((len(meta),168),np.nan,np.float32)
   ids=np.flatnonzero(meta.gap_length_h.eq(24))[:2];one=predict([data[ids[0]]],24,c,arm);two=predict([data[i] for i in ids],24,c,arm)[:1]
   np.testing.assert_allclose(one,two,atol=2e-4,rtol=2e-3)
   same=[i for i in np.flatnonzero(meta.gap_length_h.eq(24)) if len(data[i][0])==c][:2]
   if len(same)==2:
    solo=predict([data[same[0]]],24,c,arm);paired=predict([data[i] for i in same],24,c,arm)[:1]
    np.testing.assert_allclose(solo,paired,atol=2e-4,rtol=2e-3)
   if smoke:print(name,arm,'smoke passed',one.shape,flush=True);continue
   for h in gaps:
    ids=np.flatnonzero(meta.gap_length_h.eq(h))
    for st in range(0,len(ids),8):
     ii=ids[st:st+8];p=predict([data[i] for i in ii],h,c,arm)
     assert p.shape==(len(ii),h) and np.isfinite(p).all();pred[ii,:h]=p
   result=meta.copy();result['model']=name;result['arm']=arm;result['context']=c
   result['NMAE']=[float(np.mean(np.abs(pred[i,:len(a[2])]-a[2]))*a[3]) for i,a in enumerate(data)]
   result['n_eval']=result.gap_length_h;result['actual_context']=[len(a[0]) for a in data]
   result.to_csv(path,index=False);np.savez_compressed(path.with_suffix('.npz'),prediction=pred)
   print(name,arm,c,'DONE',round(time.time()-start),flush=True)
  if smoke:break
 if not smoke:(OUT/f'{name}_complete.json').write_text(json.dumps(dict(model=name,elapsed_s=time.time()-start,configurations=len(CONTEXTS)*len(arms),cases_per_configuration=len(meta),batch_independence=True),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--model');p.add_argument('--smoke',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 else:run(a.model,a.smoke)

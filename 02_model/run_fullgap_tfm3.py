"""Re-evaluate existing TimesFM3 adapters with full-gap native forecast calls.

Retains selected context, masks, reference selection and filling. Serial/batch,
hidden-truth, observation and order checks precede the fixed-mask evaluation.
"""
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import argparse,copy,hashlib,json,os,time
import numpy as np,pandas as pd,torch
import clean_protocol as cp
import run_clean_evaluation as evaluation
from clean_tfm_batch_executor import infer_many

TOL={'rtol':1e-5,'atol':1e-6}
NAMES=['TimesFM3.0','TimesFM3.0-MV','TimesFM3.0-COV','TimesFM3.0-COV-SPA']

def dump(path,obj):
 Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def block(p,r):
 return p.iloc[r.start_idx:r.end_idx+1][r.masked_vars.split(',')].to_numpy()

def poison(obj,row):
 altered=copy.copy(obj)
 for field in ['data','data_raw']:
  altered[field]=obj[field].copy()
  for variable in row.masked_vars.split(','):
   altered[field].iloc[row.start_idx:row.end_idx+1,altered[field].columns.get_loc(variable)]=1e6
 return altered

def span_audit(manifest,sites):
 rows=[]
 for r in manifest.itertuples():
  d=sites[r.greenhouse]['data']
  for v in r.masked_vars.split(','):
   missing=d[v].isna().to_numpy().copy();missing[r.start_idx:r.end_idx+1]=True
   lo,hi=int(r.start_idx),int(r.end_idx)
   while lo and missing[lo-1]:lo-=1
   while hi+1<len(missing) and missing[hi+1]:hi+=1
   rows.append(dict(case_id=int(r.case_id),variable=v,artificial_horizon=int(r.gap_length_h),merged_horizon=hi-lo+1,merged_start=lo,merged_end=hi,leading_covariate_skip=lo==0))
 return pd.DataFrame(rows)

class RecordingBackend:
 def __init__(self,backend):self.backend=backend;self.counts=Counter()
 def predict(self,context,horizon,**kw):
  self.counts[(int(horizon),'covariate' if kw.get('past_future_covariates') is not None else ('multivariate' if np.asarray(context).ndim==2 else 'univariate'))]+=1
  output=self.backend.predict(context=context,horizon=horizon,**kw)
  assert np.asarray(output.forecast).shape[-1]>=horizon, 'Incomplete native full-gap output'
  return output
 def predict_batch(self,contexts,horizon,**kw):
  for context in contexts:self.counts[(int(horizon),'covariate' if kw.get('past_future_covariates') is not None else ('multivariate' if np.asarray(context).ndim==2 else 'univariate'))]+=1
  outputs=list(self.backend.predict_batch(contexts=contexts,horizon=horizon,**kw))
  assert len(outputs)==len(contexts)
  assert all(np.asarray(o.forecast).shape[-1]>=horizon for o in outputs), 'Incomplete native full-gap output'
  return outputs

def validate(model,name,manifest,sites,folder):
 sample=manifest.groupby(['scenario','gap_length_h'],sort=False).head(1)
 examples=[(sites[r.greenhouse],r) for r in sample.itertuples()];serial=[];checks=[]
 for obj,row in examples:
  with torch.inference_mode():
   mr,p=evaluation.infer(model,name,obj,row);_,q=evaluation.infer(model,name,poison(obj,row),row)
  np.testing.assert_allclose(p.to_numpy(),q.to_numpy(),equal_nan=True,**TOL)
  observed=mr.effective_mask[cp.COLS].eq(1)
  np.testing.assert_allclose(p.values[observed.values],mr.masked_data[cp.COLS].values[observed.values],**TOL)
  serial.append((mr,p));checks.append(dict(case_id=int(row.case_id),scenario=row.scenario,gap=int(row.gap_length_h),raw_hidden_truth_poison_invariant=True,poison_max_abs=float(np.max(np.abs(block(p,row)-block(q,row))))))
  print(name,'serial/poison',row.case_id,row.scenario,row.gap_length_h,flush=True)
 batch=infer_many(model,name,examples,workers=16)
 poisoned=infer_many(model,name,[(poison(o,r),r) for o,r in examples],workers=16)
 reverse=infer_many(model,name,list(reversed(examples)),workers=16)[::-1]
 for (_,r),(_,p),(_,b),(_,q),(_,rev),check in zip(examples,serial,batch,poisoned,reverse,checks):
  for label,a,z in [('serial_batch',p,b),('batch_poison',b,q),('batch_order',b,rev)]:
   np.testing.assert_allclose(block(a,r),block(z,r),**TOL)
   check[label+'_max_abs']=float(np.max(np.abs(block(a,r)-block(z,r))))
 with torch.inference_mode():
  _,a=evaluation.infer(model,name,*examples[0]);evaluation.infer(model,name,*examples[-1]);_,b=evaluation.infer(model,name,*examples[0])
 np.testing.assert_allclose(a.to_numpy(),b.to_numpy(),equal_nan=True,**TOL)
 report=dict(model=name,forecasting_mode=model.forecasting_mode,context_len=model.context_len,raw_poison_checks=checks,ABA_order_invariance=True,serial_batch_invariant=True,batch_hidden_truth_invariant=True,batch_order_invariant=True,observations_preserved=True,tolerance=TOL,precision=torch.get_float32_matmul_precision())
 dump(folder/'validation.json',report)
 smoke=cp.OUT/'smoke'/name;smoke.mkdir(parents=True,exist_ok=True);dump(smoke/'invariance.json',report)
 return report

def validate_leading(model,name,manifest,sites,spans,folder):
 # Only covariate adapters use univariate Round-0 as the leading-gap fallback.
 if name not in ['TimesFM3.0-COV','TimesFM3.0-COV-SPA']:return
 leading=spans[spans.leading_covariate_skip]
 selected=manifest[manifest.case_id.isin(leading.case_id)]
 examples=[(sites[r.greenhouse],r) for r in selected.itertuples()]
 records=[];serial=[]
 for obj,row in examples:
  with torch.inference_mode():
   mr,p=evaluation.infer(model,name,obj,row)
   _,q=evaluation.infer(model,name,poison(obj,row),row)
  np.testing.assert_allclose(p,q,equal_nan=True,**TOL)
  safe=mr.masked_data[cp.COLS].interpolate(limit_direction='both').ffill().bfill().fillna(0.).to_numpy(np.float32)
  for r in leading[leading.case_id.eq(row.case_id)].itertuples():
   ci=cp.COLS.index(r.variable);h=int(r.merged_horizon)
   future=safe[h:h+model.context_len,ci]
   context=pd.Series(future).ffill().bfill().fillna(0.).to_numpy(np.float32)[::-1].copy()
   with torch.inference_mode():native=model._forecast_one(context,h)
   expected=native[::-1].copy();actual=p.iloc[:h,ci].to_numpy()
   np.testing.assert_allclose(actual,expected,**TOL)
   records.append(dict(case_id=int(row.case_id),variable=r.variable,merged_horizon=h,native_output_length=len(native),chronological_alignment_max_abs=float(np.abs(actual-expected).max()),hidden_truth_invariant=True))
  serial.append(p)
 batch=infer_many(model,name,examples,workers=6)
 reverse=infer_many(model,name,list(reversed(examples)),workers=6)[::-1]
 poisoned=infer_many(model,name,[(poison(o,r),r) for o,r in examples],workers=6)
 for (_,r),serial_p,(_,p),(_,q),(_,z) in zip(examples,serial,batch,reverse,poisoned):
  for a,b in [(serial_p,p),(p,q),(p,z)]:np.testing.assert_allclose(block(a,r),block(b,r),**TOL)
 with torch.inference_mode():
  _,a=evaluation.infer(model,name,*examples[0]);evaluation.infer(model,name,*examples[-1]);_,b=evaluation.infer(model,name,*examples[0])
 np.testing.assert_allclose(a,b,equal_nan=True,**TOL)
 dump(folder/'leading_alignment.json',dict(completed=True,variable_cases=len(records),mask_cases=len(examples),checks=records,serial_batch_hidden_order_invariant=True,ABA_order_invariant=True,tolerance=TOL,description='Native 1776-h forecast from reversed post-gap context is reversed back before assigning chronological positions. The final covariate stage has no past context and retains this aligned initialization.'))
 print(name,'leading 1776-h chronological alignment and invariance passed',flush=True)

def run(name,args):
 folder=cp.OUT/'fullgap_verification'/name;folder.mkdir(parents=True,exist_ok=True)
 start=time.time();torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
 original_build=evaluation.build;model=original_build(name);torch.set_float32_matmul_precision('highest')
 assert model.forecasting_mode=='full_gap' and model.context_len==1900
 model.tfm=RecordingBackend(model.tfm)
 manifest=pd.read_csv(cp.OUT/'mask_manifest.csv');sites={n:cp.site(n) for n in manifest.greenhouse.unique()}
 spans=span_audit(manifest,sites);spans.to_csv(folder/'forecast_spans.csv',index=False)
 protocol=dict(model=name,forecasting_mode='full_gap',context_len=1900,mask_jobs=len(manifest),variable_cases=len(spans),max_artificial_horizon=int(spans.artificial_horizon.max()),max_merged_horizon=int(spans.merged_horizon.max()),max_nonleading_merged_horizon=int(spans.loc[~spans.leading_covariate_skip,'merged_horizon'].max()),leading_covariate_skips=0 if name=='TimesFM3.0' else int(spans.leading_covariate_skip.sum()),extended_merged_variable_cases=int(spans.merged_horizon.ne(spans.artificial_horizon).sum()),device=torch.cuda.get_device_name(0),precision='highest',torch=str(torch.__version__),code_sha256={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in ['models/foundation_model.py','spatial.py','run_clean_evaluation.py','run_fullgap_tfm3.py']},unchanged='weights, masks, selected 1900-h context, normalization, reference selection, input filling and leading-gap fallback')
 protocol['leading_initialization_orientation']='For full-gap TimesFM only, reverse the future-context forecast back to chronological order; existing unrelated models retain their previous paths.'
 dump(folder/'protocol.json',protocol);validate(model,name,manifest,sites,folder)
 validate_leading(model,name,manifest,sites,spans,folder)
 model.tfm.counts.clear()
 evaluation.build=lambda requested,context=None:model if requested==name else original_build(requested,context)
 try:evaluation.run(name,SimpleNamespace(smoke=False,shards=1,shard=0,context=1900))
 finally:evaluation.build=original_build
 calls=[dict(horizon=h,kind=kind,calls=n) for (h,kind),n in sorted(model.tfm.counts.items())]
 dump(folder/'native_calls.json',calls)
 # Calls of 128 h would indicate accidental retained truncation; the fixed masks
 # contain no 128-h spans. Natural merged lengths remain explicitly audited.
 assert all(c['horizon']!=128 for c in calls)
 summary=json.loads((cp.OUT/'evaluation'/name/'complete.json').read_text());summary.update(verification_complete=True,elapsed_with_checks_s=time.time()-start,max_requested_horizon=max(c['horizon'] for c in calls),native_call_count=sum(c['calls'] for c in calls))
 old=json.loads((cp.OUT/'before_fullgap/evaluation'/name/'complete.json').read_text());summary['previous_NMAE']=old['NMAE'];summary['NMAE_change']=summary['NMAE']-old['NMAE'];dump(folder/'complete.json',summary)
 print('FULLGAP_DONE',json.dumps(summary),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--models',required=True);args=p.parse_args()
 for name in args.models.split(','):
  assert name in NAMES
  run(name,args)

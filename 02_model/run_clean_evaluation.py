"""Matched-mask reevaluation of all manuscript models under clean-20260911."""
from pathlib import Path
import argparse,json,time,contextlib,io,hashlib,copy
from functools import partial
import numpy as np,pandas as pd,torch
import clean_protocol as cp
import run_comparison as rc
from checkpoint import CheckpointManager
from models.imputation_models import SAITSImputation,MOMENTImputation,MOMENTFineTunedImputation
SNAP=cp.ROOT/'03_result/model_cache/huggingface/models--AutonLab--MOMENT-1-large/snapshots/ca58581bc7bea2ebed4e80dc0a3e4b8b609c6ecc'
def build(name,context=None):
 if context is None:context=1080 if name in ["Chronos2","TimesFM2.5"] else 1440
 if name=='SAITS':m=SAITSImputation(cp.OUT/'models/SAITS/best.pt')
 elif name=='MOMENT-FT':m=MOMENTFineTunedImputation(SNAP,cp.OUT/'models/MOMENT/best.pt')
 elif name=='MOMENT':m=MOMENTImputation(SNAP)
 else:
  rc._NEIGHBOR_BANK=cp.CleanNeighborBank()
  frames=[];names=[];validation=[]
  if name in rc.AG_SPECS:
   for r in pd.read_csv(cp.OUT/'sites.csv').query("group=='train'").itertuples():
    obj=cp.site(r.name);frames.append(obj['data'][cp.COLS].iloc[:obj['cut']].ffill().fillna(0.));names.append(r.name);validation.append(obj['data'][cp.COLS].iloc[obj['cut']:].ffill().fillna(0.))
  context=720 if name in rc.AG_SPECS or name in ['LI','SeasonalNaive'] else context
  models=rc.build_and_train_models([name],context,CheckpointManager(cp.OUT/'models'),frames,frames,names,600,train_paths=[Path(p) for p in json.loads((cp.OUT/'split.json').read_text())['train']],validation_frames=validation if name in rc.AG_SPECS else None);m=models[name]
  if name in rc.AG_SPECS:
   assert m.predictor is not None,'Training failure'
   m.predictor.predict=partial(m.predictor.predict,use_cache=False)
   if name=='AG-RandomForest':
    for model_name in m.predictor.model_names():
     predictor_model=m.predictor._trainer.load_model(model_name)
     if hasattr(predictor_model,'most_recent_model'):predictor_model=predictor_model.most_recent_model
     predictor_model.get_tabular_model().model.model.n_jobs=1
 for attr in ['tfm','pipeline','predictor']:
  if hasattr(m,attr) and getattr(m,attr) is None:raise RuntimeError(f'{name} {attr} unavailable')
 if hasattr(m,'_ready') and not m._ready():raise RuntimeError('Backend unavailable')
 return m

def infer(model,name,obj,row):
 mr=cp.masked_case(obj,row);cols=cp.COLS;masked=mr.masked_data[cols];eff=mr.effective_mask[cols];art=mr.artificial_mask[cols].eq(0)
 if hasattr(model,'set_greenhouse'):model.set_greenhouse(obj['name'])
 if hasattr(model,'_nb_cache'):model._nb_cache.clear()
 gs,ge=int(row.start_idx),int(row.end_idx);h=ge-gs+1
 # Explicit masks for trained reconstructors; all other methods receive only available data.
 if name in ['SAITS','MOMENT','MOMENT-FT']:
  pred=model.impute_artificial(masked,eff,art,None)
 elif name=='TimesFM3.0' or name=='TimesFM2.5' or name.startswith('AG-'):
  pred=masked.copy();ctxlen=getattr(model,'context_len',1440)
  for c in row.masked_vars.split(','):
   ctx=masked[c].iloc[max(0,gs-ctxlen):gs].interpolate(limit_direction='both').ffill().bfill().fillna(0.).to_numpy(np.float32)
   if name.startswith('AG-'):p=model._rolling_predict(c,ctx,h)
   else:
    pieces=[];left=h;cur=ctx.copy()
    while left:
     step=min(left,model.horizon_len)
     if name=='TimesFM3.0':fc=model._forecast_one(cur[-ctxlen:],step)
     else:fc=np.asarray(model.tfm.forecast(horizon=step,inputs=[cur[-ctxlen:]])[0][0])[:step]
     pieces.append(fc);cur=np.concatenate([cur,fc]);left-=step
    p=np.concatenate(pieces)
   pred.loc[pred.index[gs:ge+1],c]=p
 elif name in ['LI','SeasonalNaive','Spatial-Ridge']:
  pred=model.impute(masked,eff)
 elif name.startswith('BiTFI-'):
  pred=model.impute_artificial(masked,eff,art,None)
 elif hasattr(model,'impute_artificial'):
  safe=masked.interpolate(limit_direction='both').ffill().bfill().fillna(0.)
  pred=model.impute_artificial(masked,eff,art,safe)
 else:
  # Historical CAFI/Chronos APIs process all missing spans. Initialize natural gaps
  # from masked observations, then infer only the artificial span, within context.
  lo=max(0,gs-getattr(model,'context_len',1440));hi=min(len(masked),ge+1+getattr(model,'context_len',1440))
  working=masked.iloc[lo:hi].interpolate(limit_direction='both').ffill().bfill().fillna(0.).mask(art.iloc[lo:hi])
  am=mr.artificial_mask[cols].iloc[lo:hi]
  with contextlib.redirect_stdout(io.StringIO()):block=model.impute(working,am)
  pred=masked.copy();pred.iloc[lo:hi]=block
 for flag in ['_r0_err_reported','_cov_err_reported','_error_reported','_mv_error_reported','_art_err_reported']:
  if getattr(model,flag,False):raise RuntimeError(f'{name}: inference fell back after error ({flag})')
 assert np.isfinite(pred.values[art.values]).all(),f'{name}: nonfinite prediction'
 np.testing.assert_allclose(pred.values[eff.eq(1).values],masked.values[eff.eq(1).values],rtol=1e-5,atol=1e-6)
 return mr,pred

def score_reference(name,model,obj,row,mr,pred):
 from evaluate import compute_all_metrics
 # Reuse the existing metric adapter with a fixed prediction, not its old preprocessing.
 class Prediction:
  def impute(self,*args):return pred
 ranges={c:float(obj['data_raw'][c].max()-obj['data_raw'][c].min()) for c in cp.COLS}
 return rc._eval_one(name,Prediction(),mr,obj['scaler'],rc.make_group_labels(pred.index),obj['name'],[],cp.COLS,getattr(model,'window',getattr(model,'context_len',1440)),norm_ranges=ranges)

def score(name,model,obj,row,mr,pred):
 from clean_metrics import score_fast
 return score_fast(name,model,obj,row,mr,pred)

def run(name,args):
 folder=cp.OUT/('smoke' if args.smoke else 'evaluation')/name;folder.mkdir(parents=True,exist_ok=True)
 manifest=pd.read_csv(cp.OUT/'mask_manifest.csv')
 if args.smoke:manifest=manifest.groupby(['scenario','gap_length_h'],sort=False).head(1)
 if args.shards>1:manifest=manifest[manifest.case_id%args.shards==args.shard];folder=folder/'shards'/str(args.shard);folder.mkdir(parents=True,exist_ok=True)
 result=folder/'results.csv';done=set(pd.read_csv(result).case_id.unique()) if result.exists() else set()
 model=build(name,args.context)
 if 'TimesFM3' in name or 'TimesFM3.0' in name:torch.set_float32_matmul_precision('highest')
 sites={};start=time.time();n=0;checks=[];buffer=[]
 pending=[r for r in manifest.itertuples() if r.case_id not in done];batch_predictions={}
 for position,row in enumerate(pending):
  obj=sites.setdefault(row.greenhouse,cp.site(row.greenhouse)) if row.greenhouse not in sites else sites[row.greenhouse]
  if not args.smoke and name in ['BiTFI-TimesFM3','BiTFI-TimesFM3-fwd']:
   if row.case_id not in batch_predictions:
    from clean_batched_bitfi import infer_batch
    chunk=pending[position:position+16];examples=[]
    for r in chunk:
     if r.greenhouse not in sites:sites[r.greenhouse]=cp.site(r.greenhouse)
     examples.append((sites[r.greenhouse],r))
    outputs=infer_batch(model,examples);batch_predictions={r.case_id:value for r,value in zip(chunk,outputs)}
   mr,pred=batch_predictions.pop(row.case_id)
   art=mr.artificial_mask[cp.COLS].eq(0);assert np.isfinite(pred.values[art.values]).all()
  elif not args.smoke and name in ['TimesFM3.0','TimesFM3.0-MV','TimesFM3.0-COV','TimesFM3.0-COV-SPA','CAFI-TimesFM3-R1','CAFI-TimesFM3']:
   if row.case_id not in batch_predictions:
    from clean_tfm_batch_executor import infer_many
    chunk=pending[position:position+32];examples=[]
    for r in chunk:
     if r.greenhouse not in sites:sites[r.greenhouse]=cp.site(r.greenhouse)
     examples.append((sites[r.greenhouse],r))
    outputs=infer_many(model,name,examples);batch_predictions={r.case_id:value for r,value in zip(chunk,outputs)}
   mr,pred=batch_predictions.pop(row.case_id)
  else:
   with torch.inference_mode():mr,pred=infer(model,name,obj,row)
  if args.smoke:
   poisoned=copy.copy(obj);poisoned['data']=obj['data'].copy();poisoned['data_raw']=obj['data_raw'].copy()
   a=mr.artificial_mask.eq(0);poisoned['data_raw']=poisoned['data_raw'].mask(a,1e6)
   for c in poisoned['data']:
    good=poisoned['data_raw'][c].notna();poisoned['data'].loc[good,c]=obj['scaler'][c].transform(poisoned['data_raw'].loc[good,[c]]).ravel()
   with torch.inference_mode():_,again=infer(model,name,poisoned,row)
   np.testing.assert_allclose(pred.values,again.values,rtol=1e-5,atol=1e-6,equal_nan=True)
   checks.append(dict(case_id=int(row.case_id),scenario=row.scenario,gap=int(row.gap_length_h),raw_hidden_truth_poison_invariant=True))
  rows=score(name,model,obj,row,mr,pred)
  for r in rows:r.update(case_id=row.case_id,protocol='clean-20260911')
  buffer.extend(rows);n+=1
  if n%10==0:
   pd.DataFrame(buffer).to_csv(result,mode='a',header=not result.exists(),index=False);buffer=[]
   print(name,n,'/',len(manifest)-len(done),'jobs',round(time.time()-start,1),'seconds',flush=True)
 if buffer:pd.DataFrame(buffer).to_csv(result,mode='a',header=not result.exists(),index=False)
 if args.smoke:
  # A-B-A order test reuses the same model object, resetting job-scoped state.
  rs=list(manifest.itertuples());a=rs[0];b=rs[-1]
  with torch.inference_mode():
   _,p=infer(model,name,cp.site(a.greenhouse),a);infer(model,name,cp.site(b.greenhouse),b);_,q=infer(model,name,cp.site(a.greenhouse),a)
  np.testing.assert_allclose(p.values,q.values,rtol=1e-5,atol=1e-6,equal_nan=True)
  (folder/'invariance.json').write_text(json.dumps(dict(raw_poison_checks=checks,ABA_order_invariance=True),indent=2))
 d=pd.read_csv(result);assert set(d.case_id.unique())==set(manifest.case_id)
 a=d[d.group_type=='all'];s=a.assign(w=a.NMAE*a.n_eval).groupby('greenhouse')[['w','n_eval']].sum();scores=s.w/s.n_eval
 summary=dict(model=name,mask_jobs=len(manifest),rows=len(d),all_cells=len(a),NMAE=float(scores.mean()),SD=float(scores.std()),elapsed_s=time.time()-start,protocol='clean-20260911',inference_engine='independent-example batches (highest precision)' if name in ['BiTFI-TimesFM3','BiTFI-TimesFM3-fwd'] else ('batched backend; independent model state (highest precision)' if name in ['TimesFM3.0','TimesFM3.0-MV','TimesFM3.0-COV','TimesFM3.0-COV-SPA','CAFI-TimesFM3-R1','CAFI-TimesFM3'] and not args.smoke else 'serial'))
 (folder/'complete.json').write_text(json.dumps(summary,indent=2));print('DONE',summary,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--models',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--context',type=int,default=None);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);args=p.parse_args()
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.set_float32_matmul_precision('high')
 for name in args.models.split(','):run(name,args)

"""Untuned evaluation of common indoor variables at nine previously excluded sites."""
from pathlib import Path
import argparse,json,pickle,zlib,time,contextlib,io,copy
import numpy as np,pandas as pd,torch
import clean_protocol as cp
from experiment_paths import experiment_path,selected_context
from preprocessing import _load_file,_remove_outliers,TARGET_COLS
from masking_v2 import create_gap_masks
OUT=experiment_path('additional_sites','03_result/revision_experiments_20260926/additional_sites');V=['Tin','RH','CO2']
def prepare():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
 existing=set(pd.read_csv(cp.OUT/'sites.csv').name);split=json.loads((cp.OUT/'split.json').read_text());scalers=pickle.loads((cp.OUT/'scalers.pkl').read_bytes());rows=[];sites=[]
 for fp in split['train']:
  name=Path(fp).stem
  if name in existing:continue
  with contextlib.redirect_stdout(io.StringIO()):raw=_load_file(fp);raw=_remove_outliers(raw[[k for k,v in TARGET_COLS.items() if v in V]].rename(columns=TARGET_COLS))
  assert raw.index.is_unique and raw.index.is_monotonic_increasing
  norm=raw.copy()
  for v in V:
   finite=norm[v].notna();norm.loc[finite,v]=scalers[v].transform(norm.loc[finite,[v]]).ravel()
  obj=dict(name=name,data=norm,data_raw=raw,scaler=scalers,group='additional',cut=0)
  (OUT/'data'/f'{name}.pkl').write_bytes(pickle.dumps(obj));before=len(rows)
  for scenario,sets in [('A',[[v] for v in V]),('B',[V])]:
   for cols in sets:
    for h in [6,12,24,72,168]:
     seed=4242+zlib.crc32(f'{name}|{scenario}|{cols}|{h}'.encode())%100000
     for m in create_gap_masks(norm,norm.notna().astype('float32'),scenario,h,cols,context_len_h=selected_context('TimesFM3.0'),n_repeats=3,random_seed=seed):
      rows.append(dict(case_id=len(rows),greenhouse=name,scenario=scenario,masked_vars=','.join(cols),gap_length_h=h,repeat=m.repeat,start_idx=m.gap_start_idx,end_idx=m.gap_end_idx,start_time=str(norm.index[m.gap_start_idx])))
  sites.append(dict(name=name,rows=len(raw),cases=len(rows)-before,variables=V))
 pd.DataFrame(rows).to_csv(OUT/'mask_manifest.csv',index=False)
 (OUT/'protocol.json').write_text(json.dumps(dict(variables=V,sites=sites,existing_model_sites=sorted(existing),context=selected_context('TimesFM3.0'),scenarios=['A: one indoor sensor','B: all three indoor sensors'],gaps=[6,12,24,72,168],repeats=3,seed=4242,selection='All nine sites excluded from original scaler fitting/training/context selection because at least one outdoor target column was absent; no tuning on this cohort',normalization='Existing frozen training-prefix scalers; per-site raw range for scoring only',qualification='Additional sites from the same source, not an independently collected external cohort'),indent=2));print('PREPARED',len(rows),'jobs',len(sites),'sites',flush=True)
def evaluate():
 from run_clean_evaluation import build
 from bitfi_refinement_ablation import infer_stages
 from spatial import SpatialRidgeImputation
 torch.set_num_threads(4);torch.manual_seed(42);torch.set_float32_matmul_precision('highest')
 model=build('BiTFI-TimesFM3',selected_context('TimesFM3.0'));torch.set_float32_matmul_precision('highest');ridge=SpatialRidgeImputation(cp.CleanNeighborBank());manifest=pd.read_csv(OUT/'mask_manifest.csv');rows=list(manifest.itertuples());sites={n:pickle.loads(p.read_bytes()) for p in (OUT/'data').glob('*.pkl') for n in [p.stem]}
 depth=json.loads((experiment_path('refinement_validation/selection.json','03_result/refinement_validation_20260926/selection.json')).read_text())['selected_refinements']
 records=[];models=['BiTFI','TimesFM3-univariate','Spatial-Ridge','LI'];predictions={m:np.full((len(rows),168,3),np.nan,np.float32) for m in models};t0=time.time();checks=0
 for st in range(0,len(rows),16):
  chunk=rows[st:st+16];examples=[(sites[r.greenhouse],r) for r in chunk];batch=list(infer_stages(model,examples,max_refinements=depth,cols=V))[-1][1]
  if st==0:
   poison=[]
   for (o,r),(mr,_) in zip(examples,batch):
    q=copy.copy(o);q['data']=o['data'].mask(mr.artificial_mask.eq(0),1e6);q['data_raw']=o['data_raw'].mask(mr.artificial_mask.eq(0),1e6);poison.append((q,r))
   again=list(infer_stages(model,poison,max_refinements=depth,cols=V))[-1][1]
   for (_,p),(_,q) in zip(batch,again):np.testing.assert_allclose(p,q,rtol=1e-5,atol=1e-6,equal_nan=True);checks+=1
  for j,((o,r),(mr,bit)) in enumerate(zip(examples,batch)):
   masked=mr.masked_data[V];uni=masked.copy();li=masked.interpolate(limit_direction='both').ffill().bfill();sp=ridge.impute(masked,mr.effective_mask[V])
   for v in r.masked_vars.split(','):
    ctx=masked[v].iloc[max(0,r.start_idx-selected_context('TimesFM3.0')):r.start_idx].interpolate(limit_direction='both').ffill().bfill().fillna(0).to_numpy(np.float32)
    uni.iloc[r.start_idx:r.end_idx+1,uni.columns.get_loc(v)]=model._fc_uni(ctx,r.gap_length_h)
   for name,pred in zip(models,[bit,uni,sp,li]):
    block=pred.iloc[r.start_idx:r.end_idx+1][V].to_numpy(np.float32);predictions[name][st+j,:r.gap_length_h]=block
    for v in r.masked_vars.split(','):
     y=o['data'][v].iloc[r.start_idx:r.end_idx+1].to_numpy();p=pred[v].iloc[r.start_idx:r.end_idx+1].to_numpy();assert np.isfinite(y).all() and np.isfinite(p).all();mae=float(np.abs(y-p).mean()*o['scaler'][v].data_range_[0]);span=float(o['data_raw'][v].max()-o['data_raw'][v].min());assert span>0
     records.append(dict(case_id=int(r.case_id),greenhouse=r.greenhouse,scenario=r.scenario,masked_vars=r.masked_vars,gap_length_h=int(r.gap_length_h),variable=v,model=name,NMAE=mae/span,MAE=mae,n_eval=int(r.gap_length_h)))
  if st%64==0:print(st+len(chunk),'/',len(rows),round(time.time()-t0),flush=True)
 pd.DataFrame(records).to_csv(OUT/'results.csv',index=False)
 for name in models:np.savez_compressed(OUT/(name+'.npz'),case_id=manifest.case_id.to_numpy(),prediction=predictions[name])
 (OUT/'complete.json').write_text(json.dumps(dict(cases=len(rows),models=models,seconds=time.time()-t0,poison_checks=checks,device=torch.cuda.get_device_name(0)),indent=2));print('DONE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();prepare() if a.prepare else evaluate()

"""Supplemental evaluations and a prediction-independent Figure 6 interval."""
import argparse,json,pickle,hashlib,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd
import clean_protocol as cp
from experiment_paths import selected_context
OUT=cp.OUT

def prepare():
 from revision_additional_sites import prepare as additional_prepare
 additional_prepare()
 folder=OUT/'figure6_common_window';folder.mkdir(exist_ok=True)
 manifest=pd.read_csv(OUT/'mask_manifest.csv');eligible=[]
 for row in manifest.query("scenario=='C' and gap_length_h==72").itertuples():
  obj=cp.site(row.greenhouse);raw=obj['data_raw'][cp.COLS];lo=row.start_idx-72;hi=row.end_idx+25
  if lo>=0 and hi<=len(raw) and raw.iloc[lo:hi].notna().mean().min()>=.9:eligible.append(row)
 eligible.sort(key=lambda r:(r.greenhouse,r.start_time));r=eligible[len(eligible)//2]
 selection=dict(greenhouse=r.greenhouse,start=r.start_time,end=str(cp.site(r.greenhouse)['data'].index[r.end_idx]),repeat=int(r.repeat),manifest_case_id=int(r.case_id),eligible_cases=len(eligible),selection='Middle case after sorting eligible Scenario C 72-h cases by greenhouse identifier and start time; at least 90% raw observation coverage per variable over the 72-h preceding and 24-h following display span; no predictions or errors consulted',selected_before_example_inference=True)
 (folder/'selection.json').write_text(json.dumps(selection,indent=2))
 folder=OUT/'univariate_backbone_comparison';folder.mkdir(exist_ok=True);meta=[];records=[]
 lengths=json.loads((OUT/'context_validation/selected_contexts.json').read_text())
 # A shared length isolates backbone differences. Use the largest selected length,
 # fixed by validation, with identical available past slices for all three models.
 context=max(lengths.values())
 for r in manifest.itertuples():
  o=cp.site(r.greenhouse)
  for v in r.masked_vars.split(','):
   x=o['data'][v].iloc[max(0,r.start_idx-context):r.start_idx].interpolate(limit_direction='both').ffill().bfill().fillna(0).to_numpy(np.float32)
   y=o['data'][v].iloc[r.start_idx:r.end_idx+1].to_numpy(np.float32);factor=float(o['scaler'][v].data_range_[0]/(o['data_raw'][v].max()-o['data_raw'][v].min()))
   assert np.isfinite(x).all() and np.isfinite(y).all()
   meta.append(dict(case_id=len(meta),mask_case_id=int(r.case_id),greenhouse=r.greenhouse,scenario=r.scenario,variable=v,masked_vars=r.masked_vars,repeat=r.repeat,gap_length_h=r.gap_length_h,start_time=r.start_time,nmae_factor=factor,physical_scale=float(o['scaler'][v].data_range_[0]),group_type='all',group_value='all'))
   records.append((x,np.empty((0,len(x)+len(y)),np.float32),y,factor))
 pd.DataFrame(meta).to_csv(folder/'cases.csv',index=False);(folder/'inputs.pkl').write_bytes(pickle.dumps({context:records}))
 (folder/'protocol.json').write_text(json.dumps(dict(context_hours=context,cases=len(meta),information='Identical past-only univariate inputs; each complete gap forecast in one call',selection='Largest validation-selected context, shared across the three backbones',input_sha256=hashlib.sha256((folder/'inputs.pkl').read_bytes()).hexdigest()),indent=2))
 print(json.dumps(selection,indent=2),flush=True)

def univariate(name):
 import run_controlled_context_sweep as sweep
 folder=OUT/'univariate_backbone_comparison';context=json.loads((folder/'protocol.json').read_text())['context_hours'];sweep.OUT=folder;sweep.CONTEXTS=[context];sweep.run(name,arms=['univariate'],gaps=[6,12,24,72,168])
 d=pd.read_csv(folder/f'{name}_univariate_ctx{context}.csv');d['model']=name+'-UNI';d.to_csv(folder/f'{name}_results.csv',index=False)

def example(name):
 import torch
 from run_clean_evaluation import build,infer
 from bitfi_refinement_ablation import infer_stages
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
 folder=OUT/'figure6_common_window';sel=json.loads((folder/'selection.json').read_text());o=cp.site(sel['greenhouse']);df=o['data'];gs=df.index.get_loc(pd.Timestamp(sel['start']));ge=df.index.get_loc(pd.Timestamp(sel['end']));rows=[]
 if name=='SAITS-spatial':
  from revision_saits_spatial import ReferenceFeatures,network,OUT as train_path
  from models.imputation_models import extract_window
  m=network();m.load_state_dict(torch.load(train_path/name/'best.pt',weights_only=True)['state_dict']);m.eval();bank=ReferenceFeatures(False)
 else:m=build(name,selected_context('TimesFM3.0') if 'TimesFM3' in name else None)
 torch.set_float32_matmul_precision('highest' if 'TimesFM3' in name else 'high')
 depth=json.loads((OUT/'refinement_validation/selection.json').read_text())['selected_refinements']
 for sc,vs in [('A',[v]) for v in cp.COLS]+[('B',['Tin','RH','CO2']),('C',cp.COLS)]:
  r=SimpleNamespace(masked_vars=','.join(vs),start_idx=gs,end_idx=ge,gap_length_h=72,repeat=0,scenario=sc)
  if name=='BiTFI-TimesFM3':mr,pred=list(infer_stages(m,[(o,r)],max_refinements=depth))[-1][1][0]
  elif name=='SAITS-spatial':
   mr=cp.masked_case(o,r);masked=mr.masked_data[cp.COLS];eff=mr.effective_mask[cp.COLS];art=mr.artificial_mask[cp.COLS].eq(0);x,mask,_,lo,hi=extract_window(masked,eff,art);full=masked.to_numpy(np.float32);observed=eff.ne(0).to_numpy()&np.isfinite(full);xx,mm,_=bank.augment(o['name'],masked.index,full,observed,lo,hi,x,mask)
   with torch.inference_mode():p=m.impute({'X':torch.from_numpy(xx[None]).cuda(),'missing_mask':torch.from_numpy(mm[None].astype(np.float32)).cuda()})[0][0,:,:5].cpu().numpy()
   pred=masked.copy();block=pred.iloc[lo:hi].to_numpy(copy=True);target=art.iloc[lo:hi].to_numpy();block[target]=p[:hi-lo][target];pred.iloc[lo:hi]=block
  else:
   with torch.inference_mode():mr,pred=infer(m,name,o,r)
  lo=gs-72;hi=ge+25
  for v in vs if sc=='A' else cp.COLS:
   y=o['scaler'][v].inverse_transform(pred[v].iloc[lo:hi].to_numpy().reshape(-1,1)).ravel();rows.append(pd.DataFrame(dict(model=name,scenario=sc,variable=v,datetime=df.index[lo:hi],hours=np.arange(lo,hi)-gs,truth=o['data_raw'][v].iloc[lo:hi].to_numpy(),prediction=y,artificial=mr.artificial_mask[v].iloc[lo:hi].eq(0).to_numpy())))
 pd.concat(rows).to_csv(folder/(('BiTFI' if name=='BiTFI-TimesFM3' else name)+'.csv'),index=False)
 print('EXPORTED',name,flush=True)

def matched_availability():
 import torch
 from run_clean_evaluation import build,score
 from bitfi_refinement_ablation import infer_stages
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
 m=build('BiTFI-TimesFM3',selected_context('TimesFM3.0'));torch.set_float32_matmul_precision('highest')
 k=json.loads((OUT/'refinement_validation/selection.json').read_text())['selected_refinements']
 manifest=pd.read_csv(OUT/'mask_manifest.csv').query("scenario=='C'");folder=OUT/'matched_availability';folder.mkdir(exist_ok=True);records=[];sites={};cases=list(manifest.itertuples())
 for st in range(0,len(cases),16):
  chunk=cases[st:st+16];ex=[]
  for r in chunk:
   if r.greenhouse not in sites:sites[r.greenhouse]=cp.site(r.greenhouse)
   q=SimpleNamespace(**{key:getattr(r,key) for key in ['case_id','greenhouse','start_idx','end_idx','gap_length_h','repeat']},scenario='B',masked_vars='Tin,RH,CO2');ex.append((sites[r.greenhouse],q))
  batch=list(infer_stages(m,ex,max_refinements=k))[-1][1]
  for (o,r),(mr,pred) in zip(ex,batch):
   q=score('BiTFI-matched-B',m,o,r,mr,pred)
   for z in q:z['case_id']=r.case_id
   records.extend(q)
  print('matched availability',st+len(chunk),'/',len(cases),flush=True)
 pd.DataFrame(records).to_csv(folder/'results.csv',index=False)
 (folder/'protocol.json').write_text(json.dumps(dict(cases=len(cases),comparison='Scenario B rerun on every original Scenario C interval; compare only Tin, RH, CO2, with identical times, sites, durations and reference-selection rules',selected_refinements=k),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','univariate','example','additional','matched']);p.add_argument('--model');a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='univariate':univariate(a.model)
 elif a.action=='example':example(a.model)
 elif a.action=='matched':matched_availability()
 else:
  from revision_additional_sites import evaluate
  evaluate()

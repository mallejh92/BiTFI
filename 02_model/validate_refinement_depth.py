"""Select refinement depth using training-greenhouse validation gaps only."""
from pathlib import Path
import argparse,copy,json,time,zlib,hashlib
import numpy as np,pandas as pd,torch
import clean_protocol as cp
from masking_v2 import SCENARIO_CONFIGS
from bitfi_refinement_ablation import infer_stages
from run_clean_evaluation import build
from analyze_revision_experiments import summary,site_scores
OUT=cp.ROOT/'03_result/refinement_validation_20260926'
CANDIDATES=[1,2,3,5]
def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 assert not list(OUT.glob('shard*/complete.json')), 'Do not redefine a completed protocol'
 rows=[]
 for site in pd.read_csv(cp.OUT/'sites.csv').query("group=='train'").itertuples():
  obj=cp.site(site.name);raw=obj['data_raw'];cut=obj['cut']
  spans={v:float(raw[v].iloc[:cut].max()-raw[v].iloc[:cut].min()) for v in cp.COLS}
  for sc,sets in SCENARIO_CONFIGS.items():
   for vs in sets:
    if any(not np.isfinite(spans[v]) or spans[v]<=0 for v in vs):continue
    valid=raw[vs].notna().all(axis=1).to_numpy()
    for h in [6,12,24,72,168]:
     count=np.convolve(valid.astype(int),np.ones(h,dtype=int),'valid');starts=np.flatnonzero((count==h)&(np.arange(len(count))>=max(cut,1900)))
     rng=np.random.RandomState(42+zlib.crc32(f'{site.name}|{sc}|{vs}|{h}|refinement-validation'.encode())%100000);used=set();repeat=0
     for gs in rng.permutation(starts):
      if any(i in used for i in range(gs,gs+h)):continue
      used.update(range(gs,gs+h));rows.append(dict(case_id=len(rows),greenhouse=site.name,scenario=sc,masked_vars=','.join(vs),gap_length_h=h,repeat=repeat,start_idx=int(gs),end_idx=int(gs+h-1),start_time=str(raw.index[gs]),validation_start=cut));repeat+=1
      if repeat==3:break
 d=pd.DataFrame(rows);d.to_csv(OUT/'mask_manifest.csv',index=False)
 protocol=dict(candidates=CANDIDATES,control=0,context=1900,seed=42,repeats=3,selection='Minimum mean greenhouse NMAE across scenarios A/B/C and all five variables; hour-weighted within greenhouse, equal greenhouse weights; exact ties choose fewer passes',target_split='Final chronological 20% of training greenhouses only; test greenhouses excluded',references='Other training greenhouses only; validation target greenhouse excluded entirely from reference bank; available reference observations follow the main retrospective protocol',scaling='Existing frozen training-prefix scalers',scoring='Physical MAE divided by per-variable target-greenhouse training-prefix range',cases=len(d),sites=sorted(d.greenhouse.unique()),manifest_sha256=hashlib.sha256((OUT/'mask_manifest.csv').read_bytes()).hexdigest(),protocol_created_before_inference=True)
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol,indent=2))
def run(a):
 folder=OUT/f'shard{a.shard}';folder.mkdir(exist_ok=True)
 if (folder/'complete.json').exists():return
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
 model=build('DAFI-TimesFM3',1900);torch.set_float32_matmul_precision('highest');bank=dict(model.bank.sites)
 d=pd.read_csv(OUT/'mask_manifest.csv');d=d[d.case_id%a.shards==a.shard];scores=[];timings={k:0. for k in range(6)};checks=[];t0=time.time();done=0
 for name,g in d.groupby('greenhouse',sort=False):
  obj=cp.site(name);model.bank.sites={n:s for n,s in bank.items() if n!=name};assert name not in model.bank.sites
  rows=list(g.itertuples());spans={v:float(obj['data_raw'][v].iloc[:obj['cut']].max()-obj['data_raw'][v].iloc[:obj['cut']].min()) for v in cp.COLS}
  for st in range(0,len(rows),16):
   chunk=rows[st:st+16];examples=[(obj,r) for r in chunk];out=list(infer_stages(model,examples))
   if not checks:
    poison=[]
    for o,r in examples:
     q=copy.copy(o);q['data']=o['data'].copy();q['data_raw']=o['data_raw'].copy()
     for v in r.masked_vars.split(','):
      q['data'].iloc[r.start_idx:r.end_idx+1,q['data'].columns.get_loc(v)]=1e6;q['data_raw'].iloc[r.start_idx:r.end_idx+1,q['data_raw'].columns.get_loc(v)]=1e6
     poison.append((q,r))
    again=list(infer_stages(model,poison))
    for (_,batch,_),(_,pb,_) in zip(out,again):
     for (_,p),(_,q) in zip(batch,pb):np.testing.assert_allclose(p,q,rtol=1e-5,atol=1e-6,equal_nan=True)
    checks.append(dict(cases=len(chunk),stages=6,hidden_target_invariant=True,self_reference_excluded=True))
   for step,batch,elapsed in out:
    timings[step]+=elapsed
    if step not in [0,*CANDIDATES]:continue
    for r,(mr,pred) in zip(chunk,batch):
     assert r.start_idx>=obj['cut']
     for v in r.masked_vars.split(','):
      y=obj['data'][v].iloc[r.start_idx:r.end_idx+1].to_numpy();p=pred[v].iloc[r.start_idx:r.end_idx+1].to_numpy();assert np.isfinite(y).all() and np.isfinite(p).all()
      mae=float(np.abs(y-p).mean()*obj['scaler'][v].data_range_[0]);scores.append(dict(case_id=int(r.case_id),greenhouse=name,scenario=r.scenario,variable=v,gap_length_h=r.gap_length_h,refinements=step,model=f'BiTFI-refine{step}',n_eval=r.gap_length_h,MAE=mae,NMAE=mae/spans[v]))
   done+=len(chunk);print(a.shard,done,'/',len(d),round(time.time()-t0),'s',flush=True)
 pd.DataFrame(scores).to_csv(folder/'results.csv',index=False)
 (folder/'complete.json').write_text(json.dumps(dict(cases=len(d),seconds_per_stage=timings,elapsed_s=time.time()-t0,device=torch.cuda.get_device_name(0),precision=torch.get_float32_matmul_precision(),checks=checks),indent=2))
def select(a):
 d=pd.concat([pd.read_csv(OUT/f'shard{k}/results.csv') for k in range(a.shards)],ignore_index=True)
 assert not d.duplicated(['case_id','variable','refinements']).any()
 manifest=pd.read_csv(OUT/'mask_manifest.csv');assert d.case_id.nunique()==len(manifest)
 s=summary(d);s['refinements']=s.model.str.replace('BiTFI-refine','').astype(int);s=s.sort_values('refinements');s.to_csv(OUT/'summary.csv',index=False);site_scores(d).to_csv(OUT/'greenhouse_scores.csv',index=False);d.to_csv(OUT/'results.csv',index=False)
 chosen=int(s[s.refinements.isin(CANDIDATES)].sort_values(['mean','refinements']).iloc[0].refinements)
 result=dict(selected_refinements=chosen,selection='Minimum validation mean NMAE; exact ties favor fewer passes',summary=s.to_dict('records'),test_data_used_for_selection=False,protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest())
 (OUT/'selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--select',action='store_true');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2);a=p.parse_args()
 prepare() if a.prepare else select(a) if a.select else run(a)

"""Evaluate validation-selected refinement depth in matched control configurations."""
import argparse,json,time,copy
from pathlib import Path
import numpy as np,pandas as pd,torch
import clean_protocol as cp
from run_clean_evaluation import build,score,infer
from bitfi_refinement_ablation import infer_stages
OUT=cp.ROOT/'03_result/reevaluation_refinement_20260926'
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['DAFI-TimesFM3','DAFI-TimesFM3-fwd','DAFI-Chronos2']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--smoke',action='store_true');p.add_argument('--all-depths',action='store_true');a=p.parse_args()
 depth=5 if a.all_depths else json.loads((cp.ROOT/'03_result/refinement_validation_20260926/selection.json').read_text())['selected_refinements']
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);m=build(a.model,1900);m.refinements=depth;torch.set_float32_matmul_precision('highest')
 manifest=pd.read_csv(cp.OUT/'mask_manifest.csv')
 if a.smoke:manifest=manifest.groupby(['scenario','gap_length_h'],sort=False).head(1)
 else:manifest=manifest[manifest.case_id%a.shards==a.shard]
 folder=OUT/('smoke' if a.smoke else 'evaluation')/a.model/f'shard{a.shard}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists() and not a.smoke:return
 rows=list(manifest.itertuples());sites={};records=[];by_depth={i:[] for i in [0,1,2,3,5]};checks=[];timings={i:0. for i in range(depth+1)};t=time.time();predictions=np.full((len(rows),168,5),np.nan,np.float32)
 for st in range(0,len(rows),16):
  chunk=rows[st:st+16];ex=[]
  for r in chunk:
   if r.greenhouse not in sites:sites[r.greenhouse]=cp.site(r.greenhouse)
   ex.append((sites[r.greenhouse],r))
  stages=list(infer_stages(m,ex,max_refinements=depth))
  for k,_,dt in stages:timings[k]+=dt
  if a.smoke:
   poison=[]
   for o,r in ex:
    q=copy.copy(o);q['data']=o['data'].copy();q['data_raw']=o['data_raw'].copy()
    for v in r.masked_vars.split(','):
     q['data'].iloc[r.start_idx:r.end_idx+1,q['data'].columns.get_loc(v)]=1e6;q['data_raw'].iloc[r.start_idx:r.end_idx+1,q['data_raw'].columns.get_loc(v)]=1e6
    poison.append((q,r))
   again=list(infer_stages(m,poison,max_refinements=depth))
   for (_,batch,_),(_,pb,_) in zip(stages,again):
    for (_,pred),(_,pp) in zip(batch,pb):np.testing.assert_allclose(pred,pp,rtol=1e-5,atol=1e-6,equal_nan=True)
   restored=list(infer_stages(m,ex,max_refinements=depth))
   for (_,batch,_),(_,rb,_) in zip(stages,restored):
    for (_,pred),(_,rp) in zip(batch,rb):np.testing.assert_allclose(pred,rp,rtol=1e-5,atol=1e-6,equal_nan=True)
   checks.append(dict(cases=len(chunk),hidden_target_invariant=True,execution_order_invariant=True,depth=depth))
  if a.all_depths:
   for k,batch,_ in stages:
    if k not in by_depth:continue
    for (o,r),(mr,pred) in zip(ex,batch):
     q=score(a.model,m,o,r,mr,pred)
     for z in q:z.update(case_id=int(r.case_id),refinements=k)
     by_depth[k].extend(q)
  for i,((o,r),(mr,pred)) in enumerate(zip(ex,stages[-1][1])):
   q=score(a.model,m,o,r,mr,pred)
   for z in q:z.update(case_id=int(r.case_id),refinements=depth)
   records.extend(q);predictions[st+i,:r.gap_length_h]=pred.iloc[r.start_idx:r.end_idx+1][cp.COLS].to_numpy(np.float32)
  pd.DataFrame(records).to_csv(folder/'results.csv',index=False)
  print(a.model,st+len(chunk),'/',len(rows),round(time.time()-t),'s',flush=True)
 if a.all_depths:
  for k,rows in by_depth.items():pd.DataFrame(rows).to_csv(folder/f'refine{k}.csv',index=False)
 np.savez_compressed(folder/'predictions.npz',case_id=manifest.case_id.to_numpy(),prediction=predictions)
 (folder/'complete.json').write_text(json.dumps(dict(cases=len(rows),refinements=depth,device=torch.cuda.get_device_name(0),seconds_per_stage=timings,seconds=time.time()-t,checks=checks),indent=2))
if __name__=='__main__':main()

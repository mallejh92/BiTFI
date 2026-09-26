from pathlib import Path
import argparse,json,time,copy
import numpy as np,pandas as pd,torch
import clean_protocol as cp
from run_clean_evaluation import build,score
from bitfi_refinement_ablation import infer_stages
OUT=cp.ROOT/'03_result/revision_experiments_20260926/refinement'
def main():
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.set_float32_matmul_precision('highest')
 m=build('BiTFI-TimesFM3',1900);torch.set_float32_matmul_precision('highest');manifest=pd.read_csv(cp.OUT/'mask_manifest.csv');sites={}
 if a.smoke:manifest=manifest.groupby(['scenario','gap_length_h'],sort=False).head(1)
 else:manifest=manifest[manifest.case_id%a.shards==a.shard]
 folder=OUT/('smoke' if a.smoke else f'shard{a.shard}');folder.mkdir(parents=True,exist_ok=True)
 if (folder/'complete.json').exists():return
 rows=list(manifest.itertuples());scores={i:[] for i in [0,1,2,3,5]};preds={i:np.full((len(rows),168,5),np.nan,np.float32) for i in scores};timings={i:0. for i in range(6)};t0=time.time();checks=[]
 for st in range(0,len(rows),16):
  chunk=rows[st:st+16];examples=[]
  for r in chunk:
   if r.greenhouse not in sites:sites[r.greenhouse]=cp.site(r.greenhouse)
   examples.append((sites[r.greenhouse],r))
  out=list(infer_stages(m,examples))
  if a.smoke:
   poisoned=[]
   for (obj,r),(_,pred) in zip(examples,out[0][1]):
    q=copy.copy(obj);q['data']=obj['data'].copy();q['data_raw']=obj['data_raw'].copy()
    for v in r.masked_vars.split(','):
     q['data'].iloc[r.start_idx:r.end_idx+1,q['data'].columns.get_loc(v)]=1e6;q['data_raw'].iloc[r.start_idx:r.end_idx+1,q['data_raw'].columns.get_loc(v)]=1e6
    poisoned.append((q,r))
   again=list(infer_stages(m,poisoned))
   for (step,batch,_),(_,pb,_) in zip(out,again):
    for (_,pred),(_,pp) in zip(batch,pb):np.testing.assert_allclose(pred,pp,rtol=1e-5,atol=1e-6,equal_nan=True)
   from clean_batched_bitfi import infer_batch
   reference=infer_batch(m,examples)
   for (_,pred),(_,pp) in zip(out[1][1],reference):np.testing.assert_allclose(pred,pp,rtol=1e-5,atol=1e-6,equal_nan=True)
   checks.append(dict(cases=len(chunk),passes=6,hidden_target_invariant=True,published_one_refinement_equivalent=True))
  for step,batch,elapsed in out:
   timings[step]+=elapsed
   if step not in scores:continue
   for i,((obj,r),(mr,pred)) in enumerate(zip(examples,batch)):
    art=mr.artificial_mask[cp.COLS].eq(0);assert np.isfinite(pred.values[art.values]).all()
    q=score('BiTFI-refine'+str(step),m,obj,r,mr,pred)
    for d in q:d.update(case_id=int(r.case_id),refinements=step)
    scores[step].extend(q);preds[step][st+i,:r.gap_length_h]=pred.iloc[r.start_idx:r.end_idx+1][cp.COLS].to_numpy(np.float32)
  if st%160==0:print(a.shard,st+len(chunk),'/',len(rows),round(time.time()-t0),flush=True)
 for step in scores:
  pd.DataFrame(scores[step]).to_csv(folder/f'refine{step}.csv',index=False);np.savez_compressed(folder/f'refine{step}.npz',case_id=manifest.case_id.to_numpy(),prediction=preds[step])
 info=dict(cases=len(rows),selected_context=1900,passes=[0,1,2,3,5],seconds_per_stage=timings,elapsed_s=time.time()-t0,device=torch.cuda.get_device_name(0),torch=str(torch.__version__),matmul_precision=torch.get_float32_matmul_precision(),checks=checks)
 (folder/'complete.json').write_text(json.dumps(info,indent=2));print('DONE',info,flush=True)
if __name__=='__main__':main()

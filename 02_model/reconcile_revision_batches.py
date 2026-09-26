"""Recompute numerically batch-sensitive cases in the production batch groups."""
import argparse,json,time
import numpy as np,pandas as pd,torch
import clean_protocol as cp
from run_clean_evaluation import build,score
from bitfi_refinement_ablation import infer_stages
R=cp.ROOT/'03_result/revision_experiments_20260926';OUT=R/'refinement/production_groups'
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--shard',type=int,default=0);a=p.parse_args();OUT.mkdir(exist_ok=True)
 if a.prepare:
  d=pd.concat([pd.read_csv(R/f'refinement/shard{k}/refine1.csv') for k in range(3)]).query("group_type=='all'");o=pd.read_csv(cp.ROOT/'03_result/reevaluation_context_20260925/evaluation/BiTFI-TimesFM3/results.csv').query("group_type=='all'");z=d.merge(o,on=['case_id','variable'],suffixes=('_new','_main'));groups=sorted(set(z.loc[(z.NMAE_new-z.NMAE_main).abs()>1e-5,'case_id']//16));(OUT/'groups.json').write_text(json.dumps(dict(groups=groups,criterion='absolute case NMAE difference >1e-5; reproduce production groups of 16 consecutive cases; recompute every stage for the complete group'),indent=2));print(len(groups),'groups');return
 torch.set_num_threads(4);torch.manual_seed(42);m=build('BiTFI-TimesFM3',1900);torch.set_float32_matmul_precision('highest');manifest=pd.read_csv(cp.OUT/'mask_manifest.csv');groups=json.loads((OUT/'groups.json').read_text())['groups'][a.shard::3];sites={};records={k:[] for k in [0,1,2,3,5]};preds={k:[] for k in records};ids=[];t=time.time()
 if not groups:
  folder=OUT/f'shard{a.shard}';folder.mkdir(exist_ok=True);(folder/'complete.json').write_text(json.dumps(dict(groups=[],cases=0,seconds=0,matmul_precision='highest')));return
 for g in groups:
  rows=list(manifest.iloc[g*16:(g+1)*16].itertuples());ex=[]
  for r in rows:
   if r.greenhouse not in sites:sites[r.greenhouse]=cp.site(r.greenhouse)
   ex.append((sites[r.greenhouse],r))
  for step,batch,elapsed in infer_stages(m,ex):
   if step not in records:continue
   for (obj,r),(mr,pred) in zip(ex,batch):
    q=score('BiTFI-refine'+str(step),m,obj,r,mr,pred)
    for z in q:z.update(case_id=int(r.case_id),refinements=step)
    records[step].extend(q);block=np.full((168,5),np.nan,np.float32);block[:r.gap_length_h]=pred.iloc[r.start_idx:r.end_idx+1][cp.COLS];preds[step].append(block)
  ids.extend([int(r.case_id) for r in rows]);print(g,'done',round(time.time()-t),flush=True)
 folder=OUT/f'shard{a.shard}';folder.mkdir(exist_ok=True)
 for k in records:
  pd.DataFrame(records[k]).to_csv(folder/f'refine{k}.csv',index=False);np.savez_compressed(folder/f'refine{k}.npz',case_id=np.array(ids),prediction=np.stack(preds[k]))
 (folder/'complete.json').write_text(json.dumps(dict(groups=groups,cases=len(ids),seconds=time.time()-t,device=torch.cuda.get_device_name(0),matmul_precision=torch.get_float32_matmul_precision()),indent=2));print('DONE')
if __name__=='__main__':main()

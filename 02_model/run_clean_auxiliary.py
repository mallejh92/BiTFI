"""Reevaluate context/round sensitivity with raw-observation masks."""
import argparse,json,time,contextlib,io
from pathlib import Path
import pandas as pd,numpy as np,torch
import clean_protocol as cp
from run_clean_evaluation import build,infer,score
p=argparse.ArgumentParser();p.add_argument('--family',choices=['tfm3','chronos'],required=True);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
manifest=pd.read_csv(cp.OUT/'mask_manifest.csv');sites={r.name:cp.site(r.name) for r in pd.read_csv(cp.OUT/'sites.csv').query("group=='test'").itertuples()}
models=['TimesFM3.0','TimesFM3.0-MV','TimesFM3.0-COV'] if a.family=='tfm3' else ['Chronos2','TimesFM2.5','CAFI']
folder='context_sweep_tfm3' if a.family=='tfm3' else 'context_sweep';subset=manifest[(manifest.scenario=='A')&manifest.gap_length_h.isin([24,168])&(manifest.repeat<3)]
for context in [96,168,336,720,1080,1440,1900]:
 out=cp.OUT/folder/f'ctx{context}';out.mkdir(parents=True,exist_ok=True)
 for name in models:
  path=out/f'{name}.csv'
  if path.exists():continue
  m=build(name,context);rows=[];start=time.time()
  items=list(subset.itertuples())
  if a.family=='tfm3':
   from clean_tfm_batch_executor import infer_many
   torch.set_float32_matmul_precision('highest')
   for start_i in range(0,len(items),32):
    chunk=items[start_i:start_i+32];examples=[(sites[r.greenhouse],r) for r in chunk]
    outputs=infer_many(m,name,examples)
    for r,(mr,pred) in zip(chunk,outputs):rows.extend(score(name,m,sites[r.greenhouse],r,mr,pred))
  else:
   for r in items:
    obj=sites[r.greenhouse]
    with torch.inference_mode():mr,pred=infer(m,name,obj,r)
    rows.extend(score(name,m,obj,r,mr,pred))
  pd.DataFrame(rows).to_csv(path,index=False);print('context',context,name,'DONE',round(time.time()-start),flush=True)
 pd.concat([pd.read_csv(out/f'{name}.csv') for name in models]).to_csv(out/'results.csv',index=False)
# Fixed explicit R0..R5 sensitivity, same A-independent B/C mask subset across backbones.
name='CAFI-TimesFM3' if a.family=='tfm3' else 'CAFI';out=cp.OUT/'round_sweep';out.mkdir(exist_ok=True);path=out/f'rounds_{a.family}.csv'
if not path.exists():
 m=build(name);m.tol=-1.;m.var_tol=-1.;captured=[];r0=m._round0_impute;cr=m._covariate_round
 def wrap(fn):
  def f(*args,**kwargs):
   value=fn(*args,**kwargs);captured.append(value.copy());return value
  return f
 m._round0_impute=wrap(r0);m._covariate_round=wrap(cr);rows=[]
 subset=manifest[manifest.scenario.isin(['B','C'])&manifest.gap_length_h.isin([24,168])&(manifest.repeat<3)]
 for r in subset.itertuples():
  captured.clear();obj=sites[r.greenhouse]
  with torch.inference_mode():mr,pred=infer(m,name,obj,r)
  assert len(captured)==6
  idx=obj['data'].index[int(r.start_idx):int(r.end_idx)+1]
  for rnd,estimate in enumerate(captured):
   estimate=m._inverse_transform_frame(estimate)
   for v in r.masked_vars.split(','):
    true=obj['data'].loc[idx,v];y=estimate.loc[idx,v];rows.append(dict(backbone=a.family,greenhouse=r.greenhouse,scenario=r.scenario,gap_length_h=r.gap_length_h,repeat=r.repeat,variable=v,**{'round':rnd},mae_norm=float(np.mean(np.abs(true-y)))))
 pd.DataFrame(rows).to_csv(path,index=False)
(cp.OUT/f'{a.family}_auxiliary_complete.json').write_text(json.dumps({'protocol':'clean-20260911','context_masks':'Fixed A/24,168 h, first three repeats from master manifest; same masks across context limits','round_masks':'Fixed B/C 24,168 h first three repeats; force R0..R5 without early stop'},indent=2))
print('AUXILIARY COMPLETE',a.family,flush=True)

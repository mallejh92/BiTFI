"""Independent processes for slow models; preserve the exact per-case adapter."""
import sys,subprocess,time,json,os
from pathlib import Path
import pandas as pd
import clean_protocol as cp
name=sys.argv[1];count=int(sys.argv[2]);folder=cp.OUT/'evaluation'/name;full=folder/'results.csv'
if full.exists():
 d=pd.read_csv(full)
 for i in range(count):
  p=folder/'shards'/str(i);p.mkdir(parents=True,exist_ok=True);d[d.case_id%count==i].to_csv(p/'results.csv',index=False)
 full.rename(folder/'results_serial_prefix.csv')
jobs=[];handles=[];start=time.time();gpus=os.environ.get('BITFI_SHARD_GPUS',os.environ.get('CUDA_VISIBLE_DEVICES','0')).split(',')
for i in range(count):
 h=(cp.OUT/'logs'/f'{name}_shard{i}.log').open('a');handles.append(h)
 jobs.append(subprocess.Popen([sys.executable,'-u',str(cp.ROOT/'02_model/run_clean_evaluation.py'),'--models',name,'--shard',str(i),'--shards',str(count)],stdout=h,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus[i%len(gpus)]}))
codes=[p.wait() for p in jobs]
for h in handles:h.close()
assert all(c==0 for c in codes),codes
frames=[]
for i in range(count):
 p=folder/'shards'/str(i);assert (p/'complete.json').exists();frames.append(pd.read_csv(p/'results.csv'))
d=pd.concat(frames,ignore_index=True);manifest=pd.read_csv(cp.OUT/'mask_manifest.csv');assert set(d.case_id)==set(manifest.case_id)
assert not d.duplicated(['case_id','group_type','group_value','variable']).any()
d.sort_values(['case_id','group_type','group_value','variable']).to_csv(full,index=False)
a=d[d.group_type=='all'];s=a.assign(w=a.NMAE*a.n_eval).groupby('greenhouse')[['w','n_eval']].sum();scores=s.w/s.n_eval
report=dict(model=name,mask_jobs=len(manifest),rows=len(d),all_cells=len(a),NMAE=float(scores.mean()),SD=float(scores.std()),elapsed_s=time.time()-start,protocol='clean-20260911',inference_engine=f'{count} independent processes; unchanged model adapter; independent backend batching for TimesFM3; includes saved prefix')
(folder/'complete.json').write_text(json.dumps(report,indent=2));print('DONE',report,flush=True)

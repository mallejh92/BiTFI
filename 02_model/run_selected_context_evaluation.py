"""Reevaluate foundation-model configurations with validation-selected contexts.

Isolated output; training checkpoints, original test masks and unchanged baseline
results are reused with explicit provenance. No old result is overwritten.
"""
import argparse,json,shutil,os,sys,runpy
from pathlib import Path
from types import SimpleNamespace
import clean_protocol as cp
OLD=cp.OUT;OUT=cp.ROOT/'03_result/reevaluation_context_20260925'
SELECTION=cp.ROOT/'03_result/context_validation_20260925/selected_contexts.json'
FAMILIES={'Chronos2':'Chronos2','DAFI-Chronos2':'Chronos2','TimesFM2.5':'TimesFM2.5',**{m:'TimesFM3.0' for m in ['TimesFM3.0','TimesFM3.0-MV','TimesFM3.0-COV','TimesFM3.0-COV-SPA','DAFI-TimesFM3','DAFI-TimesFM3-fwd']}}
def prepare():
 OUT.mkdir(exist_ok=True)
 for n in ['data','models','sites.csv','scalers.pkl','split.json','mask_manifest.csv','univariate_backbone_comparison']:
  if not (OUT/n).exists():(OUT/n).symlink_to(OLD/n,target_is_directory=(OLD/n).is_dir())
 for n in ['evaluation','smoke','logs','figure6_common_window']:(OUT/n).mkdir(exist_ok=True)
 for d in (OLD/'evaluation').iterdir():
  if d.is_dir() and d.name not in FAMILIES and not (OUT/'evaluation'/d.name).exists():(OUT/'evaluation'/d.name).symlink_to(d,target_is_directory=True)
 for n in ['SAITS','MOMENT-FT','Spatial-Ridge']:
  p=OUT/'figure6_common_window'/f'{n}.csv'
  if not p.exists():p.symlink_to(OLD/'figure6_common_window'/f'{n}.csv')
 s=json.loads(SELECTION.read_text());p=json.loads((OLD/'protocol.json').read_text());p.update(version='validation-context-20260925',base_protocol=str(OLD),selection_source=str(SELECTION),selected_contexts=s,model_contexts={m:s[f] for m,f in FAMILIES.items()},unchanged_baselines='Existing results reused; parameters, masks, checkpoints and inputs unchanged',S6_control='Matched past-only univariate contexts up to validation-selected 1900 h; full test mask manifest, each gap in one call')
 (OUT/'protocol.json').write_text(json.dumps(p,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--model');p.add_argument('--smoke',action='store_true');p.add_argument('--figure6',action='store_true');a=p.parse_args()
 if a.prepare:prepare();return
 # New calls serialize per configuration; also respect workers already launched.
 import fcntl,time
 stage='smoke' if a.smoke else 'evaluation'
 folder=OUT/stage/a.model;folder.mkdir(parents=True,exist_ok=True)
 lock=(folder/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
 if not a.figure6:
  marker=folder/'legacy_running.pid'
  if marker.exists():
   pid=int(marker.read_text())
   while pid!=os.getpid() and Path(f'/proc/{pid}').exists() and not (folder/'complete.json').exists():time.sleep(10)
  if (folder/'complete.json').exists():return
 import torch
 torch.set_num_threads(4);torch.manual_seed(42)
 import numpy as np
 np.random.seed(42);torch.set_float32_matmul_precision('highest' if 'TimesFM3' in a.model else 'high')
 cp.OUT=OUT
 import run_clean_evaluation as ev
 context=json.loads(SELECTION.read_text())[FAMILIES[a.model]]
 if a.figure6:
  original=ev.build;ev.build=lambda name,context_arg=None:original(name,context)
  sys.argv=['export_clean_figure6.py','--model',a.model];runpy.run_path(str(cp.ROOT/'02_model/export_clean_figure6.py'),run_name='__main__');return
 ev.run(a.model,SimpleNamespace(smoke=a.smoke,context=context,shards=1,shard=0))
 folder=OUT/('smoke' if a.smoke else 'evaluation')/a.model
 p=json.loads((folder/'complete.json').read_text());p.update(protocol='validation-context-20260925',selected_context=context,selection_source=str(SELECTION));(folder/'complete.json').write_text(json.dumps(p,indent=2))
if __name__=='__main__':main()

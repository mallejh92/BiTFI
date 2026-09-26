"""Reproduce the existing comparison on a complete hourly time grid.

Set BITFI_RUN_ROOT to an empty output directory. This entry point never copies
predictions or trained checkpoints from a different time-axis protocol.
"""
import argparse, json, os, sys, runpy
from pathlib import Path
os.environ.setdefault('BITFI_RUN_ROOT',str(Path(__file__).resolve().parents[1]/'03_result/reevaluation_hourly_20260927'))
os.environ.setdefault('BITFI_CLEAN','1')
os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parents[1]/'03_result/model_cache/huggingface'))
os.environ.setdefault('JAX_PLATFORMS','cpu')
import clean_protocol as cp
from experiment_paths import selected_context
FAMILIES={'Chronos2':'Chronos2','BiTFI-Chronos2':'Chronos2','TimesFM2.5':'TimesFM2.5',**{m:'TimesFM3.0' for m in ['TimesFM3.0','TimesFM3.0-MV','TimesFM3.0-COV','TimesFM3.0-COV-SPA','BiTFI-TimesFM3','BiTFI-TimesFM3-fwd']}}

def execute(script,arguments):
 sys.argv=[script,*arguments];runpy.run_path(str(cp.ROOT/'02_model'/script),run_name='__main__')

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['windows','context-prepare','context','context-select','train','saits-prepare','saits-evaluate','evaluate','depth-prepare','depth','depth-select','refinement','controls','auxiliary']);p.add_argument('--model');p.add_argument('--aux-action');p.add_argument('--smoke',action='store_true');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
 if a.action=='auxiliary':execute('hourly_auxiliary.py',[a.aux_action]+(['--model',a.model] if a.model else []))
 elif a.action=='windows':cp.windows(cp.OUT/'models/SAITS',512)
 elif a.action.startswith('context'):
  execute('run_validation_context_selection.py', ['--prepare'] if a.action=='context-prepare' else ['--select'] if a.action=='context-select' else ['--model',a.model])
 elif a.action=='train':
  if a.model=='SAITS':execute('train_saits.py',['--clean','--out',str(cp.OUT/'models/SAITS')])
  elif a.model=='MOMENT-FT':execute('train_moment_head.py',[])
  else:execute('revision_saits_spatial.py',['--train','spatial' if a.model=='SAITS-spatial' else 'local'])
 elif a.action=='saits-prepare':execute('revision_saits_spatial.py',['--prepare'])
 elif a.action=='saits-evaluate':execute('revision_saits_spatial.py',['--evaluate'])
 elif a.action=='evaluate':
  from types import SimpleNamespace
  import numpy as np,torch
  import run_clean_evaluation as ev
  torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
  args=SimpleNamespace(smoke=a.smoke,context=selected_context(FAMILIES[a.model]) if a.model in FAMILIES else None,shard=a.shard,shards=a.shards)
  ev.run(a.model,args)
 elif a.action.startswith('depth'):
  args=['--prepare'] if a.action=='depth-prepare' else ['--select','--shards',str(a.shards)] if a.action=='depth-select' else ['--shard',str(a.shard),'--shards',str(a.shards)]
  execute('validate_refinement_depth.py',args)
 elif a.action=='refinement':execute('run_revision_refinement.py',['--shard',str(a.shard),'--shards',str(a.shards)]+(['--smoke'] if a.smoke else []))
 elif a.action=='controls':execute('run_selected_refinement_controls.py',['--model',a.model,'--shard',str(a.shard),'--shards',str(a.shards)]+(['--smoke'] if a.smoke else []))
if __name__=='__main__':main()

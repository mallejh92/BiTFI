import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'03_result/reevaluation_context_20260925'
for model in sys.argv[1:]:
 for stage in ['smoke','evaluation']:
  if (out/stage/model/'complete.json').exists():continue
  cmd=[sys.executable,'-u',str(ROOT/'02_model/run_selected_context_evaluation.py'),'--model',model]+(['--smoke'] if stage=='smoke' else [])
  with (out/'logs'/f'{model}_{stage}.log').open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
  print(model,stage,'complete',flush=True)
 if model=='BiTFI-TimesFM3':
  subprocess.run([sys.executable,'-u',str(ROOT/'02_model/run_selected_context_evaluation.py'),'--model',model,'--figure6'],check=True,cwd=ROOT)
print('QUEUE COMPLETE',flush=True)

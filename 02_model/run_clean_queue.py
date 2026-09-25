"""Sequential smoke-then-full execution; aborts on any failed validation."""
import subprocess,sys,json,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'03_result/reevaluation_clean_20260911';(out/'logs').mkdir(exist_ok=True)
wait_for=os.environ.get("BITFI_WAIT_FOR")
while wait_for and not (out/"evaluation"/wait_for/"complete.json").exists():time.sleep(20)
for model in sys.argv[1:]:
 for stage in ['smoke','evaluation']:
  if (out/stage/model/'complete.json').exists():continue
  cmd=[sys.executable,'-u',str(ROOT/'02_model/run_clean_evaluation.py'),'--models',model]+(['--smoke'] if stage=='smoke' else [])
  print(time.strftime('%H:%M:%S'),model,stage,flush=True)
  with (out/'logs'/f'{model}_{stage}.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
print('QUEUE COMPLETE',flush=True)

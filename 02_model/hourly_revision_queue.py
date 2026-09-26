"""Dependency-aware local queue for a complete hourly reproduction."""
import os,json,sys,time,subprocess,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=Path(os.environ.get('BITFI_RUN_ROOT',ROOT/'03_result/reevaluation_hourly_20260927'))
def main():
 queue=Path(sys.argv[1]);jobs=json.loads(queue.read_text());folder=RUN/'logs';folder.mkdir(parents=True,exist_ok=True)
 with (folder/(queue.stem+'.lock')).open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for j in jobs:
   done=folder/(j['name']+'.done.json')
   if done.exists():continue
   for path in j.get('wait',[]):
    while not (RUN/path).exists():time.sleep(10)
   env=os.environ.copy();env.update(BITFI_RUN_ROOT=str(RUN),CUDA_VISIBLE_DEVICES=str(j['gpu']),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
   command=[str(ROOT/j.get('env','.venv')/'bin/python'),'-u',str(ROOT/'02_model/hourly_revision.py'),*j['args']]
   print(time.strftime('%H:%M:%S'),j['name'],flush=True);start=time.time()
   (folder/(queue.stem+'.status.json')).write_text(json.dumps(dict(task=j['name'],start=start,command=command)))
   with (folder/(j['name']+'.log')).open('a') as log:
    result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   status=dict(task=j['name'],returncode=result.returncode,elapsed_s=time.time()-start)
   if result.returncode:
    (folder/(j['name']+'.failed.json')).write_text(json.dumps(status));raise RuntimeError(status)
   done.write_text(json.dumps(status))
 print('QUEUE COMPLETE',queue.stem,flush=True)
if __name__=='__main__':main()

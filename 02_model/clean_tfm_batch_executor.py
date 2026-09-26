"""Batch only backend calls; retain each model's exact Python imputation logic.
Workers own separate model state. One dispatcher serializes access to the backend.
"""
import copy,queue,threading,time,sys
from concurrent.futures import Future,ThreadPoolExecutor
from collections import defaultdict
import numpy as np,torch
class Dispatcher:
 def __init__(self,backend):
  self.backend=backend;self.q=queue.Queue();self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
 def submit(self,context,horizon,**kwargs):
  f=Future();self.q.put((context,horizon,kwargs,f));return f
 def _run(self):
  while True:
   first=self.q.get()
   if first is None:return
   items=[first];deadline=time.monotonic()+.004
   while len(items)<32:
    try:items.append(self.q.get(timeout=max(.0001,deadline-time.monotonic())))
    except queue.Empty:break
    if time.monotonic()>=deadline:break
   groups=defaultdict(list)
   for item in items:
    ctx,h,kw,f=item;pf=kw.get('past_future_covariates');po=kw.get('past_only_covariates')
    shape=np.asarray(ctx).shape
    key=(h,len(shape),shape[0] if len(shape)>1 else 1,None if pf is None else np.atleast_2d(pf).shape[0],None if po is None else np.atleast_2d(po).shape[0],tuple(sorted((k,v) for k,v in kw.items() if k not in ['past_future_covariates','past_only_covariates'])),shape[-1])
    groups[key].append(item)
   for key,group in groups.items():
    kw=dict(group[0][2]);kw.pop('past_future_covariates',None);kw.pop('past_only_covariates',None)
    if key[3] is not None:kw['past_future_covariates']=[x[2]['past_future_covariates'] for x in group]
    if key[4] is not None:kw['past_only_covariates']=[x[2]['past_only_covariates'] for x in group]
    try:
     with torch.inference_mode():outputs=list(self.backend.predict_batch(contexts=[x[0] for x in group],horizon=key[0],**kw))
     if len(outputs)!=len(group):raise RuntimeError('Incomplete batched response')
     for item,value in zip(group,outputs):item[3].set_result(value)
    except BaseException as error:
     for item in group:item[3].set_exception(error)
 def close(self):self.q.put(None);self.thread.join()
class Proxy:
 def __init__(self,dispatcher):self.dispatcher=dispatcher
 def predict(self,context,horizon,**kwargs):return self.dispatcher.submit(context,horizon,**kwargs).result()
 def predict_batch(self,contexts,horizon,**kwargs):
  requests=[]
  for i,ctx in enumerate(contexts):
   kw={k:(v[i] if k in ['past_future_covariates','past_only_covariates'] and v is not None else v) for k,v in kwargs.items()}
   requests.append(self.dispatcher.submit(ctx,horizon,**kw))
  for f in requests:yield f.result()
def infer_many(model,name,examples,workers=32):
 from run_clean_evaluation import infer
 original_stdout=sys.stdout
 dispatcher=Dispatcher(model.tfm)
 bank_lock=threading.Lock()
 class LockedBank:
  def select(self,*args,**kwargs):
   with bank_lock:return model.bank.select(*args,**kwargs)
 def job(example):
  instance=copy.copy(model);instance.tfm=Proxy(dispatcher)
  if hasattr(instance,'_nb_cache'):instance._nb_cache={}
  if hasattr(instance,'bank'):instance.bank=LockedBank()
  if hasattr(instance,'stats'):instance.stats=copy.deepcopy(instance.stats)
  with torch.inference_mode():return infer(instance,name,*example)
 try:
  with ThreadPoolExecutor(max_workers=workers) as pool:return list(pool.map(job,examples))
 finally:
  dispatcher.close();sys.stdout=original_stdout

"""Matched SAITS information controls; validation values never enter training references."""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np,pandas as pd,torch
from torch.utils.data import Dataset,DataLoader
import clean_protocol as cp
from models.imputation_models import extract_window,WINDOW
from train_saits import corrupt
from dafi import DAFIImputation
OUT=cp.ROOT/'03_result/revision_experiments_20260926/saits_information'
NFEATURES=24

class ReferenceFeatures:
 def __init__(self,training=False):
  self.training=training;self.cache={};self.sites={}
  for r in pd.read_csv(cp.OUT/'sites.csv').query("group=='train'").itertuples():
   o=cp.site(r.name);self.sites[r.name]=o['data'][cp.COLS].iloc[:o['cut']] if training else o['data'][cp.COLS]
 def candidates(self,name,index,v):
  key=(name,v,len(index),str(index[0]),str(index[-1]))
  if key not in self.cache:
   names=[];cols=[]
   for n,d in self.sites.items():
    if n==name:continue
    a=d[v].reindex(index).to_numpy(np.float64)
    if np.isfinite(a).mean()<.8:continue
    names.append(n);cols.append(a)
   a=np.stack(cols) if cols else np.empty((0,len(index)))
   self.cache[key]=(names,a,np.isfinite(a),np.nan_to_num(a,nan=0.))
  return self.cache[key]
 def select(self,name,index,v,target,observed):
  names,a,valid,z=self.candidates(name,index,v);fit=observed&np.isfinite(target)
  if fit.sum()<500 or not len(names):return np.zeros((len(index),3),np.float32),np.zeros((len(index),3),np.uint8),[]
  good=valid&fit[None,:];n=good.sum(axis=1);x=np.where(good,z,0.);y=np.where(good,np.nan_to_num(target,nan=0.)[None,:],0.)
  sx=x.sum(1);sy=y.sum(1);den=np.maximum(n,1)
  cov=(x*y).sum(1)-sx*sy/den;varx=(x*x).sum(1)-sx*sx/den;vary=(y*y).sum(1)-sy*sy/den
  with np.errstate(divide='ignore',invalid='ignore'):r=cov/np.sqrt(np.maximum(varx,0)*np.maximum(vary,0))
  ii=sorted([j for j in range(len(names)) if n[j]>=500 and np.isfinite(r[j])],key=lambda j:(abs(r[j]),names[j]),reverse=True)[:3]
  out=np.zeros((len(index),3),np.float32);mask=np.zeros_like(out,np.uint8)
  for j,k in enumerate(ii):out[:,j]=pd.Series(a[k]).interpolate(limit_direction='both').ffill().bfill().to_numpy(np.float32);mask[:,j]=1
  return out,mask,[names[k] for k in ii]
 def augment(self,name,index,full,observed,lo,hi,x,mask):
  out=np.zeros((512,NFEATURES),np.float32);m=np.zeros_like(out,np.uint8);length=hi-lo
  out[:,:5]=x;m[:,:5]=mask
  out[:length,5:9]=DAFIImputation._time_cov(index[lo:hi]).T;m[:length,5:9]=1
  selected={}
  for j,v in enumerate(cp.COLS):
   nb,nm,names=self.select(name,index,v,full[:,j],observed[:,j]);out[:length,9+3*j:12+3*j]=nb[lo:hi];m[:length,9+3*j:12+3*j]=nm[lo:hi];selected[v]=names
  return out,m,selected

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);banks={True:ReferenceFeatures(True),False:ReferenceFeatures(False)};sizes={};t0=time.time()
 for split in ['train','validation']:
  xs=[];ms=[];ys=[];hs=[];metadata=[];idx=0
  for r in pd.read_csv(cp.OUT/'sites.csv').query("group=='train'").itertuples():
   obj=cp.site(r.name);frame=obj['data'][cp.COLS];cut=obj['cut'];a=frame.to_numpy(np.float32)
   left,right,stride=(0,cut,128) if split=='train' else (cut,len(a),256)
   for st in range(left,right-512+1,stride):
    raw=a[st:st+512]
    if np.isfinite(raw).mean()<.8 or np.isfinite(raw).sum(0).min()<48:continue
    for rep in range(4 if split=='train' else 1):
     seed=42+idx*997+rep if split=='train' else 142+idx
     b=corrupt(raw,np.random.RandomState(seed));full=a[:cut].copy() if split=='train' else a.copy();observed=np.isfinite(full);observed[st:st+512]&=b['missing_mask'].astype(bool)
     # No hidden target value is ever available to reference ranking.
     full[~observed]=np.nan;index=frame.index[:cut] if split=='train' else frame.index
     x,m,names=banks[split=='train'].augment(r.name,index,full,observed,st,st+512,b['X'],b['missing_mask'])
     assert all(r.name not in ns for ns in names.values())
     xs.append(x);ms.append(m);ys.append(b['X_holdout']);hs.append(b['indicating_mask'].astype(np.uint8));metadata.append(dict(greenhouse=r.name,start=st,repeat=rep,seed=seed,reference_sites=names))
    idx+=1
  np.savez_compressed(OUT/(split+'.npz'),X=np.stack(xs),mask=np.stack(ms),truth=np.stack(ys),holdout=np.stack(hs));(OUT/(split+'_cases.json')).write_text(json.dumps(metadata));sizes[split]=dict(windows=idx,corruptions=len(xs));print(split,sizes[split],round(time.time()-t0),flush=True)
 info=dict(window=512,features=['five local sensors','four calendar features','three same-variable references per target variable'],n_features=24,training_reference_values='training prefixes only, target greenhouse excluded',reference_ranking='masked target only; at least 500 paired observations and 80% candidate temporal coverage',sizes=sizes,model_architecture='official SAITS, unchanged attention blocks; d_feature=24',objective='ORT averaged across three estimates + MIT, both scored only on five local target variables',selection='minimum validation masked MAE, fixed validation masks; no test checkpoint selection',training=dict(seed=42,max_epochs=60,patience=10,batch_size=16,learning_rate=.001),context_difference='512-h window versus main BiTFI 1900-h sides; information control is not an equal-context architecture test')
 (OUT/'protocol.json').write_text(json.dumps(info,indent=2))

def network():
 import sys
 sys.path.insert(0,str(cp.ROOT/'third_party/SAITS'));from modeling.saits import SAITS
 return SAITS(n_groups=2,n_group_inner_layers=1,d_time=512,d_feature=24,d_model=256,d_inner=128,n_head=4,d_k=64,d_v=64,dropout=.1,input_with_mask=True,param_sharing_strategy='inner_group',MIT=True,diagonal_attention_mask=True,device='cuda').cuda()
class Arrays(Dataset):
 def __init__(self,split,spatial):
  d=np.load(OUT/(split+'.npz'));self.x=d['X'];self.mask=d['mask'];self.truth=d['truth'];self.h=d['holdout']
  if not spatial:self.x[:,:,9:]=0;self.mask[:,:,9:]=0
 def __len__(self):return len(self.x)
 def __getitem__(self,i):return [torch.from_numpy(a[i].astype(np.float32)) for a in [self.x,self.mask,self.truth,self.h]]

def train(spatial):
 name='SAITS-spatial' if spatial else 'SAITS-matched-local';folder=OUT/name;folder.mkdir(exist_ok=True)
 if (folder/'complete.json').exists():return
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.set_float32_matmul_precision('high')
 tr=DataLoader(Arrays('train',spatial),batch_size=16,shuffle=True,num_workers=0);va=DataLoader(Arrays('validation',spatial),batch_size=16)
 model=network();opt=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');stale=0;history=[];t0=time.time()
 for epoch in range(1,61):
  model.train();losses=[]
  for b in tr:
   x,m,y,h=[v.cuda() for v in b];opt.zero_grad(set_to_none=True);_,est=model.impute({'X':x,'missing_mask':m});ort=sum(((z[:,:,:5]-x[:,:,:5]).abs()*m[:,:,:5]).sum()/(m[:,:,:5].sum()+1e-9) for z in est)/3;mit=((est[-1][:,:,:5]-y).abs()*h).sum()/(h.sum()+1e-9);loss=ort+mit
   assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss))
  model.eval();total=0.;count=0.
  with torch.inference_mode():
   for b in va:
    x,m,y,h=[v.cuda() for v in b];pred=model.impute({'X':x,'missing_mask':m})[0][:,:,:5];total+=float(((pred-y).abs()*h).sum());count+=float(h.sum())
  metric=total/count;history.append(dict(epoch=epoch,train_loss=np.mean(losses),validation_mae=metric,elapsed_s=time.time()-t0));pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
  if metric<best:
   best=metric;stale=0;torch.save(dict(state_dict=model.state_dict(),epoch=epoch,window=512,n_features=24,spatial=spatial,validation_mae=best),folder/'best.pt')
  else:stale+=1
  print(name,epoch,round(metric,6),'best',round(best,6),'seconds',round(time.time()-t0),flush=True)
  if stale>=10:break
 (folder/'complete.json').write_text(json.dumps(dict(best_validation_mae=best,epochs=epoch,elapsed_s=time.time()-t0,device=torch.cuda.get_device_name(0),torch=str(torch.__version__)),indent=2))

def evaluate():
 torch.set_num_threads(4);torch.set_float32_matmul_precision("high")
 from run_clean_evaluation import score
 folder=OUT/'evaluation';folder.mkdir(exist_ok=True);manifest=pd.read_csv(cp.OUT/'mask_manifest.csv');bank=ReferenceFeatures(False);models={};scores={};preds={};sites={};checks=[];t0=time.time()
 for name in ['SAITS-matched-local','SAITS-spatial']:
  model=network();model.load_state_dict(torch.load(OUT/name/'best.pt',weights_only=True)['state_dict']);model.eval();models[name]=model;scores[name]=[];preds[name]=np.full((len(manifest),168,5),np.nan,np.float32)
 for st in range(0,len(manifest),32):
  cases=list(manifest.iloc[st:st+32].itertuples());features=[];metadata=[]
  for r in cases:
   if r.greenhouse not in sites:sites[r.greenhouse]=cp.site(r.greenhouse)
   o=sites[r.greenhouse];mr=cp.masked_case(o,r);masked=mr.masked_data[cp.COLS];eff=mr.effective_mask[cp.COLS];art=mr.artificial_mask[cp.COLS].eq(0);x,m,_,lo,hi=extract_window(masked,eff,art);full=masked.to_numpy(np.float32);observed=eff.ne(0).to_numpy()&np.isfinite(full)
   xx,mm,names=bank.augment(r.greenhouse,masked.index,full,observed,lo,hi,x,m)
   if st==0:
    poison=full.copy();poison[~observed]=1e6;px,pm,_=bank.augment(r.greenhouse,masked.index,poison,observed,lo,hi,x,m);np.testing.assert_array_equal(xx,px);np.testing.assert_array_equal(mm,pm);checks.append(int(r.case_id))
   features.append((xx,mm));metadata.append((o,r,mr,masked,art,lo,hi))
  for name,model in models.items():
   x=np.stack([z[0] for z in features]);m=np.stack([z[1] for z in features])
   if name=='SAITS-matched-local':x[:,:,9:]=0;m[:,:,9:]=0
   with torch.inference_mode():p=model.impute({'X':torch.from_numpy(x).cuda(),'missing_mask':torch.from_numpy(m.astype(np.float32)).cuda()})[0][:,:,:5].cpu().numpy()
   for j,(o,r,mr,masked,art,lo,hi) in enumerate(metadata):
    pred=masked.copy();block=pred.iloc[lo:hi].to_numpy(copy=True);target=art.iloc[lo:hi].to_numpy();block[target]=p[j,:hi-lo][target];pred.iloc[lo:hi]=block
    assert np.isfinite(pred.to_numpy()[art]).all();adapter=type('Adapter',(),{'window':512})();q=score(name,adapter,o,r,mr,pred)
    for z in q:z['case_id']=int(r.case_id)
    scores[name].extend(q);preds[name][st+j,:r.gap_length_h]=pred.iloc[r.start_idx:r.end_idx+1][cp.COLS]
  if st%320==0:print('evaluation',st+len(cases),'/',len(manifest),round(time.time()-t0),flush=True)
 for name in models:
  pd.DataFrame(scores[name]).to_csv(folder/(name+'.csv'),index=False);np.savez_compressed(folder/(name+'.npz'),case_id=manifest.case_id.to_numpy(),prediction=preds[name])
 (folder/'complete.json').write_text(json.dumps(dict(cases=len(manifest),models=list(models),elapsed_s=time.time()-t0,feature_poison_checks=len(checks),device=torch.cuda.get_device_name(0)),indent=2));print('EVALUATION COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--train',choices=['local','spatial']);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.train:train(a.train=='spatial')
 else:evaluate()

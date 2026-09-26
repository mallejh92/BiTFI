"""Hourly-grid evaluation protocol with separated training and evaluation inputs.
Raw observation masks; train-prefix-only fixed scalers; mask-before-fill.
"""
from pathlib import Path
import json,pickle,zlib,contextlib,io,hashlib
import numpy as np,pandas as pd
from sklearn.preprocessing import MinMaxScaler
from preprocessing import _load_file,_select_and_rename,_remove_outliers,TARGET_COLS
from experiment_paths import experiment_path
from preprocessing import assert_hourly
from masking_v2 import create_gap_masks,SCENARIO_CONFIGS
ROOT=Path(__file__).resolve().parents[1];OUT=experiment_path('', '03_result/reevaluation_clean_20260911');COLS=list(TARGET_COLS.values())
PROTOCOL='hourly-20260927'
def prepare():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
 split=json.loads((ROOT/'03_result/comparison/split.json').read_text());(OUT/'split.json').write_text(json.dumps(split,indent=2));raw={};groups={}
 for group in ['train','test']:
  for fp in split[group]:
   with contextlib.redirect_stdout(io.StringIO()):
    d=_select_and_rename(_load_file(fp),include_covariates=True)
   if d is None:continue
   d=_remove_outliers(d);assert_hourly(d.index);name=Path(fp).stem;raw[name]=d;groups[name]=group
   print('raw',group,name,flush=True)
 assert not ({Path(p).stem for p in split['train']}&{Path(p).stem for p in split['test']})
 scaler={}
 for c in sorted(set().union(*(set(d.columns) for d in raw.values()))):
  xs=[d[c].iloc[:int(.8*len(d))].dropna().values for n,d in raw.items() if groups[n]=='train' and c in d]
  x=np.concatenate(xs);scaler[c]=MinMaxScaler().fit(x.reshape(-1,1))
 with (OUT/'scalers.pkl').open('wb') as f:pickle.dump(scaler,f)
 manifest=[];sites=[]
 for name,d in raw.items():
  norm=d.copy()
  for c in norm:
   finite=norm[c].notna();norm.loc[finite,c]=scaler[c].transform(norm.loc[finite,[c]]).ravel()
  obj=dict(name=name,data=norm,data_raw=d,scaler=scaler,group=groups[name],cut=int(.8*len(d)))
  with (OUT/'data'/f'{name}.pkl').open('wb') as f:pickle.dump(obj,f)
  sites.append(dict(name=name,group=groups[name],rows=len(d),source_rows=d.attrs.get('source_rows',len(d)),inserted_hourly_rows=d.attrs.get('inserted_hourly_rows',0),split_index=obj['cut'],validation_start=str(d.index[obj['cut']])))
  if groups[name]!='test':continue
  valid=norm.notna().astype('float32')
  for sc,combos in SCENARIO_CONFIGS.items():
   for vs in combos:
    for h in [6,12,24,72,168]:
     seed=42+zlib.crc32(f'{name}{sc}{h}'.encode())%9999
     for m in create_gap_masks(norm,valid,sc,h,vs,context_len_h=1440,n_repeats=10,random_seed=seed):
      manifest.append(dict(case_id=len(manifest),greenhouse=name,scenario=sc,masked_vars=','.join(vs),gap_length_h=h,repeat=m.repeat,start_idx=m.gap_start_idx,end_idx=m.gap_end_idx,start_time=str(norm.index[m.gap_start_idx]),end_time=str(norm.index[m.gap_end_idx])))
 pd.DataFrame(manifest).to_csv(OUT/'mask_manifest.csv',index=False);pd.DataFrame(sites).to_csv(OUT/'sites.csv',index=False)
 protocol=dict(version=PROTOCOL,time_axis='One hour per position; absent timestamps are NaN; duplicate and off-hour records rejected; first 80% of hourly recording span defines training prefix',normalization='One fixed scaler per variable fitted exclusively to first 80% of training greenhouses; no test/validation targets used',raw_validity='Physical-range filtering only, no short-gap interpolation before masking',natural_missing='Fill only from masked input, per independent job; imputers may retain explicit missingness masks',neighbors='Training greenhouses only, selected afresh for each mask using available target observations',evaluation='Raw observed targets only; fixed full raw site range used only for scoring',mask_context=1440,n_mask_jobs=len(manifest),seed=42,split_sha256=hashlib.sha256((OUT/'split.json').read_bytes()).hexdigest())
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print(protocol,flush=True)
def site(name):
 with (OUT/'data'/f'{name}.pkl').open('rb') as f:return pickle.load(f)
def windows(out,window):
 train=[];val=[];sites=[]
 for r in pd.read_csv(OUT/'sites.csv').query("group=='train'").itertuples():
  obj=site(r.name);a=obj['data'][COLS].to_numpy(np.float32);counts=[]
  for bucket,left,right,stride in [(train,0,obj['cut'],128),(val,obj['cut'],len(a),256)]:
   before=len(bucket)
   for st in range(left,right-window+1,stride):
    block=a[st:st+window]
    if np.isfinite(block).mean()>=.8 and np.isfinite(block).sum(axis=0).min()>=48:bucket.append(block)
   counts.append(len(bucket)-before)
  sites.append(dict(name=r.name,split_index=obj['cut'],n_train=counts[0],n_val=counts[1]))
 train=np.stack(train);val=np.stack(val);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'windows.npz',train=train,val=val)
 info=dict(sites=sites,window=window,train_windows=len(train),validation_windows=len(val),split='Chronological 80/20 within training sites, no crossing windows; no pre-split interpolation',scaling='Fixed training-prefix-only scalers; natural missingness retained',masking='Train/validation corruption applied to genuine observations only',protocol=PROTOCOL)
 (out/'training_protocol.json').write_text(json.dumps(info,indent=2));return train,val,info
class CleanNeighborBank:
 def __new__(cls):
  from spatial import NeighborBank
  obj=object.__new__(NeighborBank);obj.target_vars=COLS;obj.sites={r.name:site(r.name)['data'][COLS] for r in pd.read_csv(OUT/'sites.csv').query("group=='train'").itertuples()};return obj

def masked_case(obj,row):
 from masking_v2 import MaskResult
 full=obj['data'];assert_hourly(full.index);assert full.index[int(row.end_idx)]-full.index[int(row.start_idx)]==pd.Timedelta(hours=int(row.gap_length_h)-1);art=pd.DataFrame(1.,index=full.index,columns=full.columns);vs=row.masked_vars.split(',');art.loc[full.index[int(row.start_idx):int(row.end_idx)+1],vs]=0.
 valid=full.notna().astype('float32');masked=full.mask(art.eq(0));eff=valid*art
 return MaskResult(masked,art,eff,full.copy(),art.eq(0)&valid.eq(1),vs,int(row.start_idx),int(row.end_idx),int(row.repeat),row.scenario,int(row.gap_length_h),1440)
if __name__=='__main__':prepare()

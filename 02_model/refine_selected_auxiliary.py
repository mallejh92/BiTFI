"""Apply selected refinement depth to additional sites and the fixed example."""
from pathlib import Path
from types import SimpleNamespace
import json,pickle,time,copy
import numpy as np,pandas as pd,torch
import clean_protocol as cp
from run_clean_evaluation import build
from bitfi_refinement_ablation import infer_stages
ROOT=cp.ROOT;OLD=ROOT/'03_result/revision_experiments_20260926/additional_sites';OUT=ROOT/'03_result/reevaluation_refinement_20260926'
def main():
 k=json.loads((ROOT/'03_result/refinement_validation_20260926/selection.json').read_text())['selected_refinements']
 if k==1:return
 torch.set_num_threads(4);torch.manual_seed(42);m=build('BiTFI-TimesFM3',1900);m.refinements=k;torch.set_float32_matmul_precision('highest');t=time.time();folder=OUT/'additional_sites';folder.mkdir(exist_ok=True);cols=['Tin','RH','CO2'];manifest=pd.read_csv(OLD/'mask_manifest.csv');sites={n:pickle.loads(p.read_bytes()) for p in (OLD/'data').glob('*.pkl') for n in [p.stem]};rows=list(manifest.itertuples());records=[];checks=[]
 for st in range(0,len(rows),16):
  chunk=rows[st:st+16];ex=[(sites[r.greenhouse],r) for r in chunk];stages=list(infer_stages(m,ex,max_refinements=k,cols=cols));batch=stages[-1][1]
  if st==0:
   poison=[]
   for (o,r),(mr,_) in zip(ex,batch):
    q=copy.copy(o);q['data']=o['data'].mask(mr.artificial_mask.eq(0),1e6);q['data_raw']=o['data_raw'].mask(mr.artificial_mask.eq(0),1e6);poison.append((q,r))
   again=list(infer_stages(m,poison,max_refinements=k,cols=cols))[-1][1]
   for (_,p),(_,q) in zip(batch,again):np.testing.assert_allclose(p,q,rtol=1e-5,atol=1e-6,equal_nan=True)
   checks.append(dict(cases=len(chunk),hidden_target_invariant=True,refinements=k))
  for (o,r),(mr,pred) in zip(ex,batch):
   for v in r.masked_vars.split(','):
    y=o['data'][v].iloc[r.start_idx:r.end_idx+1].to_numpy();p=pred[v].iloc[r.start_idx:r.end_idx+1].to_numpy();assert np.isfinite(p).all();mae=float(np.abs(y-p).mean()*o['scaler'][v].data_range_[0]);span=float(o['data_raw'][v].max()-o['data_raw'][v].min());records.append(dict(case_id=int(r.case_id),greenhouse=r.greenhouse,scenario=r.scenario,masked_vars=r.masked_vars,gap_length_h=int(r.gap_length_h),variable=v,model='BiTFI',NMAE=mae/span,MAE=mae,n_eval=int(r.gap_length_h)))
  print('additional',st+len(chunk),'/',len(rows),round(time.time()-t),'s',flush=True)
 original=pd.read_csv(OLD/'results.csv');pd.concat([original[original.model!='BiTFI'],pd.DataFrame(records)],ignore_index=True).to_csv(folder/'results.csv',index=False)
 (folder/'complete.json').write_text(json.dumps(dict(cases=len(rows),selected_refinements=k,seconds=time.time()-t,device=torch.cuda.get_device_name(0),checks=checks),indent=2))
 selection=json.loads((ROOT/'03_result/reevaluation_context_20260925/figure6_common_window/selection.json').read_text());o=cp.site(selection['greenhouse']);df=o['data'];gs=df.index.get_loc(pd.Timestamp(selection['start']));ge=df.index.get_loc(pd.Timestamp(selection['end']));ex=[]
 for sc,sets in [('A',[[v] for v in cp.COLS]),('B',[['Tin','RH','CO2']]),('C',[cp.COLS])]:
  for vs in sets:ex.append((o,SimpleNamespace(masked_vars=','.join(vs),start_idx=gs,end_idx=ge,gap_length_h=72,repeat=selection['repeat'],scenario=sc)))
 batch=list(infer_stages(m,ex,max_refinements=k))[-1][1];out=[]
 for (o,r),(mr,pred) in zip(ex,batch):
  lo=max(0,gs-72);hi=min(len(df),ge+25)
  for v in (r.masked_vars.split(',') if r.scenario=='A' else cp.COLS):
   y=o['scaler'][v].inverse_transform(pred[v].iloc[lo:hi].to_numpy().reshape(-1,1)).ravel();out.append(pd.DataFrame(dict(model='BiTFI-TimesFM3',scenario=r.scenario,variable=v,datetime=df.index[lo:hi],hours=np.arange(lo,hi)-gs,truth=o['data_raw'][v].iloc[lo:hi].to_numpy(),prediction=y,artificial=mr.artificial_mask[v].iloc[lo:hi].eq(0).to_numpy())))
 pd.concat(out).to_csv(OUT/'figure6_common_window/BiTFI.csv',index=False);print('Exported selected-depth example')
if __name__=='__main__':main()

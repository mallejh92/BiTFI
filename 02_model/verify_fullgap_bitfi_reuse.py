"""Verify unchanged BiTFI predictions after full-gap adapter revisions."""
from pathlib import Path
import argparse,copy,hashlib,json,time
import numpy as np,pandas as pd,torch
import clean_protocol as cp
from run_clean_evaluation import build,infer_bitfi_batch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--previous-code',type=Path);args=p.parse_args()
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
 folder=cp.OUT/'fullgap_verification/BiTFI-reuse';folder.mkdir(parents=True,exist_ok=True)
 manifest=pd.read_csv(cp.OUT/'mask_manifest.csv')
 mf=manifest.groupby(['scenario','gap_length_h'],sort=False).head(1)
 spans=cp.OUT/'fullgap_verification/TimesFM3.0-COV-SPA/forecast_spans.csv'
 if spans.exists():
  d=pd.read_csv(spans);leading=manifest[manifest.case_id.isin(d.loc[d.leading_covariate_skip,'case_id'])]
  mf=pd.concat([mf,leading]).drop_duplicates('case_id')
 sites={n:cp.site(n) for n in mf.greenhouse.unique()};examples=[(sites[r.greenhouse],r) for r in mf.itertuples()]
 path=cp.OUT/'evaluation/BiTFI-TimesFM3/predictions.npz';before=sha(path);old=np.load(path);old={int(k):v for k,v in zip(old['case_id'],old['prediction'])}
 model=build('BiTFI-TimesFM3');torch.set_float32_matmul_precision('highest');assert model.refinements==5 and model.context_len==1900
 start=time.time();fresh=infer_bitfi_batch(model,examples)
 corrupted=[]
 for obj,row in examples:
  q=copy.copy(obj)
  for field in ['data','data_raw']:
   q[field]=obj[field].copy()
   for v in row.masked_vars.split(','):q[field].iloc[row.start_idx:row.end_idx+1,q[field].columns.get_loc(v)]=1e6
  corrupted.append((q,row))
 poison=infer_bitfi_batch(model,corrupted);reversed_batch=infer_bitfi_batch(model,list(reversed(examples)))[::-1];checks=[]
 for (_,r),(mr,a),(_,b),(_,c) in zip(examples,fresh,poison,reversed_batch):
  indices=[cp.COLS.index(v) for v in r.masked_vars.split(',')];x=a.iloc[r.start_idx:r.end_idx+1][cp.COLS].to_numpy()[:,indices];y=old[r.case_id][:r.gap_length_h][:,indices]
  np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-6)
  np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6,equal_nan=True)
  np.testing.assert_allclose(a,c,rtol=1e-5,atol=1e-6,equal_nan=True)
  obs=mr.effective_mask[cp.COLS].eq(1)
  np.testing.assert_allclose(a.values[obs.values],mr.masked_data[cp.COLS].values[obs.values],rtol=1e-5,atol=1e-6)
  checks.append(dict(case_id=int(r.case_id),scenario=r.scenario,gap=int(r.gap_length_h),stored_masked_max_abs=float(np.abs(x-y).max()),hidden_truth_max_abs=float(np.nanmax(np.abs(a.to_numpy()-b.to_numpy()))),batch_reverse_max_abs=float(np.nanmax(np.abs(a.to_numpy()-c.to_numpy())))))
 files=['bitfi.py','bitfi_refinement_ablation.py','clean_protocol.py','clean_tfm_batch_executor.py']
 unchanged={f:dict(current_sha256=sha(Path(__file__).parent/f)) for f in files}
 if args.previous_code:
  for f in files:
   unchanged[f]['previous_sha256']=sha(args.previous_code/f);unchanged[f]['unchanged']=unchanged[f]['current_sha256']==unchanged[f]['previous_sha256'];assert unchanged[f]['unchanged']
 result=dict(completed=True,cases=len(checks),checks=checks,original_tolerance=dict(rtol=1e-5,atol=1e-6),selected_context=1900,selected_refinements=5,device=torch.cuda.get_device_name(0),matmul_precision='highest',stored_prediction_sha256=before,stored_predictions_unmodified=before==sha(path),core_files=unchanged,elapsed_s=time.time()-start)
 (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

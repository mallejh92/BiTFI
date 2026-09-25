"""Verify identical algorithms separately from GPU/batch numeric reproducibility."""
import json,copy
import numpy as np,pandas as pd,torch
import clean_protocol as cp
from run_clean_evaluation import build
from bitfi_refinement_ablation import infer_stages
from clean_batched_bitfi import infer_batch
from clean_metrics import score_fast
OUT=cp.ROOT/'03_result/revision_experiments_20260926/analysis'
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.manual_seed(42)
model=build('DAFI-TimesFM3',1900);torch.set_float32_matmul_precision('highest');manifest=pd.read_csv(cp.OUT/'mask_manifest.csv');ids=[2833,2780,2744,2909,2930,2878,227,2681,2805,2743,2998,2819,2754,2753,2914];examples=[(cp.site(r.greenhouse),r) for r in manifest[manifest.case_id.isin(ids)].itertuples()]
a=list(infer_stages(model,examples,max_refinements=1))[-1][1];b=infer_batch(model,examples);checks=[]
for (obj,r),(_,p),(_,q) in zip(examples,a,b):
 np.testing.assert_allclose(p,q,rtol=1e-5,atol=1e-6,equal_nan=True);checks.append(dict(case_id=int(r.case_id),max_prediction_diff=float(np.nanmax(np.abs(p.to_numpy()-q.to_numpy())))))
# Reproduce production grouping for the largest discrepancy, on original hardware.
# Production evaluation processes consecutive chunks of 16 cases.
records=[]
for start in [int(i//16*16) for i in [2780,2833]]:
 ex=[(cp.site(r.greenhouse),r) for r in manifest.iloc[start:start+16].itertuples()]
 for (obj,r),(mr,pred) in zip(ex,infer_batch(model,ex)):
  for z in score_fast('check',model,obj,r,mr,pred):z['case_id']=int(r.case_id);records.append(z)
pd.DataFrame(records).to_csv(OUT/'production_group_recheck.csv',index=False)
(OUT/'algorithm_equivalence.json').write_text(json.dumps(dict(device=torch.cuda.get_device_name(0),cases=checks),indent=2));print('Algorithm equivalence passed',checks)

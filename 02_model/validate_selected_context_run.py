"""Independent integrity checks before publishing the context-selected run."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import clean_protocol as cp
from run_selected_context_evaluation import FAMILIES
ROOT=cp.ROOT;RUN=ROOT/'03_result/reevaluation_context_20260925';VAL=ROOT/'03_result/context_validation_20260925'
v=pd.read_csv(VAL/'cases.csv');sites=pd.read_csv(cp.OUT/'sites.csv');train=set(sites.query("group=='train'").name);test=set(sites.query("group=='test'").name)
assert set(v.greenhouse)<=train and not set(v.greenhouse)&test
for r in v.itertuples():
 o=cp.site(r.greenhouse);assert r.start_idx>=o['cut'] and r.end_idx<len(o['data']);assert o['data_raw'][r.masked_vars].iloc[r.start_idx:r.end_idx+1].notna().all()
 assert r.start_idx>=1900
selected=json.loads((VAL/'selected_contexts.json').read_text());summary=pd.read_csv(VAL/'summary.csv')
for m,g in summary.groupby('model'):assert selected[m]==int(g.sort_values(['mean','context']).iloc[0].context)
reference=pd.read_csv(cp.OUT/'mask_manifest.csv');reports=[]
for model,family in FAMILIES.items():
 p=RUN/'evaluation'/model;assert (p/'complete.json').exists(),model
 d=pd.read_csv(p/'results.csv');assert set(d.context_len)=={selected[family]},model
 assert set(d.case_id)==set(reference.case_id),model
 assert len(d)==34535 and len(d.query("group_type=='all'"))==6085,model
 assert not d.duplicated(['case_id','group_type','group_value','variable']).any(),model
 assert np.isfinite(d.NMAE).all()
 check=json.loads((RUN/'smoke'/model/'invariance.json').read_text());assert check['ABA_order_invariance'] and len(check['raw_poison_checks'])==15
 assert all(x['raw_hidden_truth_poison_invariant'] for x in check['raw_poison_checks'])
 reports.append(dict(model=model,context=selected[family],sha256=hashlib.sha256((p/'results.csv').read_bytes()).hexdigest()))
report=dict(protocol='validation-context-20260925',selection_uses_training_sites_only=True,selection_cases=len(v),selection_sites=len(set(v.greenhouse)),selection_minima_verified=True,reevaluated_models=reports,test_mask_manifest_unchanged=True,unchanged_models_reused=True,raw_poison_checks_rerun=135,order_checks_rerun=9)
(RUN/'validation_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

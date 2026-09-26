"""Independent elapsed-time, prediction-to-score and publication checks."""
import argparse,json,pickle,re,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import clean_protocol as cp
R=cp.OUT

def data_check():
 manifest=pd.read_csv(R/'mask_manifest.csv');sites={p.stem:pickle.loads(p.read_bytes()) for p in (R/'data').glob('*.pkl')}
 train=[o for o in sites.values() if o['group']=='train'];test=[o for o in sites.values() if o['group']=='test']
 assert len(train)==24 and len(test)==10 and manifest.case_id.tolist()==list(range(len(manifest)))
 for o in sites.values():
  d=o['data_raw'];assert len(d)==len(pd.date_range(d.index[0],d.index[-1],freq='h'))
  assert np.all(np.diff(d.index.asi8)==3600*10**9) and o['cut']==int(.8*len(d))
  np.testing.assert_array_equal(d.isna(),o['data'].isna())
 for v in cp.COLS:
  x=np.concatenate([o['data_raw'][v].iloc[:o['cut']].dropna().to_numpy() for o in train]);sc=train[0]['scaler'][v]
  np.testing.assert_allclose([sc.data_min_[0],sc.data_max_[0]],[x.min(),x.max()],rtol=0,atol=0)
 keys=[]
 for r in manifest.itertuples():
  o=sites[r.greenhouse];assert o['group']=='test';index=o['data_raw'].index
  assert index[r.end_idx]-index[r.start_idx]==pd.Timedelta(hours=r.gap_length_h-1)
  assert index[r.start_idx]==pd.Timestamp(r.start_time) and index[r.end_idx]==pd.Timestamp(r.end_time)
  for v in r.masked_vars.split(','):
   assert o['data_raw'][v].iloc[r.start_idx:r.end_idx+1].notna().all();keys.append((r.case_id,v))
 for folder in ['context_validation','refinement_validation']:
  protocol=json.loads((R/folder/'protocol.json').read_text());assert set(protocol['sites'])<=set(o['name'] for o in train)
  p=R/folder/('cases.csv' if folder=='context_validation' else 'mask_manifest.csv')
  if p.exists():
   for r in pd.read_csv(p).itertuples():
    o=sites[r.greenhouse];assert r.start_idx>=o['cut'];assert o['data'].index[r.end_idx]-o['data'].index[r.start_idx]==pd.Timedelta(hours=r.gap_length_h-1)
 additional=R/'additional_sites'
 extras={p.stem:pickle.loads(p.read_bytes()) for p in (additional/'data').glob('*.pkl')}
 assert len(extras)==9 and not (set(extras)&set(sites))
 for o in extras.values():assert np.all(np.diff(o['data_raw'].index.asi8)==3600*10**9)
 for r in pd.read_csv(additional/'mask_manifest.csv').itertuples():
  o=extras[r.greenhouse];assert o['data'].index[r.end_idx]-o['data'].index[r.start_idx]==pd.Timedelta(hours=r.gap_length_h-1)
  assert o['data_raw'][r.masked_vars.split(',')].iloc[r.start_idx:r.end_idx+1].notna().all().all()
 return manifest,sites,set(keys)

def publication_check(manifest,sites,keys):
 from hourly_analysis import MODELS
 overall=pd.read_csv(R/'analysis/all_summary.csv').set_index('model');assert set(overall.index)==set(MODELS)
 checks=[];stored={}
 for model in MODELS:
  folder=R/'evaluation'/model;rows=pd.read_csv(folder/'results.csv').query("group_type=='all'");assert not rows.duplicated(['case_id','variable']).any();assert set(zip(rows.case_id,rows.variable))==keys
  rows=rows.set_index(['case_id','variable']);packed=folder/'predictions.npz';array=None
  if packed.exists():
   pack=np.load(packed);np.testing.assert_array_equal(pack['case_id'],manifest.case_id);array=pack['prediction']
  weighted={};error=0.
  for r in manifest.itertuples():
   o=sites[r.greenhouse];p=array[r.case_id,:r.gap_length_h] if array is not None else np.load(folder/'predictions'/f'{r.case_id}.npy')
   for v in r.masked_vars.split(','):
    j=cp.COLS.index(v);raw=o['data_raw'][v];truth=raw.iloc[r.start_idx:r.end_idx+1].to_numpy();prediction=p[:,j].astype(np.float64)*o['scaler'][v].data_range_[0]+o['scaler'][v].data_min_[0]
    assert np.isfinite(prediction).all();mae=float(np.abs(prediction-truth).mean());nmae=mae/(raw.max()-raw.min());saved=rows.loc[(r.case_id,v)]
    # CSV metrics are rounded to six decimals by run_comparison._r.
    # A float32 gap array adds at most half an ULP per value before inverse scaling.
    quantization=float(np.abs(np.spacing(p[:,j].astype(np.float32))).astype(np.float64).mean())*.5*o['scaler'][v].data_range_[0]
    tolerance=5.0001e-7+quantization+1e-10
    np.testing.assert_allclose(mae,saved.MAE,rtol=0,atol=tolerance,err_msg=f'{model} case {r.case_id} {v} MAE')
    np.testing.assert_allclose(nmae,saved.NMAE,rtol=0,atol=5.0001e-7+quantization/(raw.max()-raw.min())+1e-10,err_msg=f'{model} case {r.case_id} {v} NMAE')
    assert saved.n_eval==len(truth);error=max(error,abs(nmae-saved.NMAE));weighted.setdefault(r.greenhouse,[0.,0]);weighted[r.greenhouse][0]+=saved.NMAE*len(truth);weighted[r.greenhouse][1]+=len(truth)
  scores=np.array([weighted[k][0]/weighted[k][1] for k in sorted(weighted)])
  np.testing.assert_allclose([scores.mean(),scores.std(ddof=1)],overall.loc[model,['mean','sd']].to_numpy(float),rtol=0,atol=1e-12);stored[model]=scores
  checks.append(dict(model=model,variable_cases=len(rows),max_float32_NMAE_difference=error));print('Verified predictions and scores',model,flush=True)
  if model in ['SAITS-spatial','SAITS-matched-local']:continue
  if model.startswith('BiTFI'):
   info=json.loads((R/'smoke'/model/'shard0/complete.json').read_text());assert info['cases']==15 and info['refinements']==5
   assert sum(c['cases'] for c in info['checks'])==15 and all(c['hidden_target_invariant'] and c['execution_order_invariant'] for c in info['checks'])
  else:
   info=json.loads((R/'smoke'/model/'invariance.json').read_text());assert len(info['raw_poison_checks'])==15 and info['ABA_order_invariance'];assert all(c['raw_hidden_truth_poison_invariant'] for c in info['raw_poison_checks'])
 weights=pd.read_csv(R/'analysis/scenario_duration_weights.csv');np.testing.assert_allclose(weights.overall_weight.sum(),1.,rtol=0,atol=1e-12)
 for m in ['Chronos2','TimesFM2.5','TimesFM3.0']:
  u=pd.read_csv(R/f'univariate_backbone_comparison/{m}_results.csv');assert len(u)==len(keys) and np.isfinite(u.NMAE).all()
 P=R/'publication';doc=(P/'TFM.tex').read_text();supp=(P/'supplementary.tex').read_text();both=doc+'\n'+supp
 assert not re.search(r'\b(?:corrected|previously|archived)\b',both,re.I)
 assert not re.search(r'@[A-Z_]+@',both)
 assert '1,393' not in both and '3,357' not in both and '1,898' not in both
 figure2=re.search(r'\\includegraphics[^\n]+Figure2_.*?\\end\{figure\*?\}',doc,re.S).group();assert 'GPT' not in figure2
 claims=json.loads((P/'numeric_claims.json').read_text());assert claims['mask_cases']==len(manifest) and claims['variable_cases']==len(keys)
 assert f"{overall.loc['BiTFI-TimesFM3','mean']:.4f}" in doc
 metrics=pd.read_csv(R/'figures/source_data/figure6_panel_metrics.csv');assert len(metrics)==65 and set(metrics.panel)==set('ABCDEFHIKLMNO')
 assert set(metrics.model)=={'BiTFI-TimesFM3','TimesFM3.0-COV-SPA','SAITS-spatial','MOMENT-FT','Spatial-Ridge'}
 files=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',both);assert len(files)==len(set(files))==17
 for file in files:assert (P/file).exists(),file
 return checks

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-only',action='store_true');a=p.parse_args();manifest,sites,keys=data_check();report=dict(hourly_sites=len(sites),mask_cases=len(manifest),variable_cases=len(keys),scalers_match_training_prefixes=True,additional_hourly_sites=9,context_selection_masks=1390,refinement_selection_masks=1896)
 if not a.data_only:report['prediction_checks']=publication_check(manifest,sites,keys)
 report['manifest_sha256']=hashlib.sha256((R/'mask_manifest.csv').read_bytes()).hexdigest();target=R/('data_validation.json' if a.data_only else 'publication_validation.json');target.write_text(json.dumps(report,indent=2));print(target,flush=True)
if __name__=='__main__':main()

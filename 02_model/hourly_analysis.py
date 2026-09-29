"""Publishable statistics derived exclusively from one hourly run."""
import json,shutil,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import clean_protocol as cp
from analyze_revision_experiments import summary,site_scores,compare
from revision_config import MAIN_MODELS
OUT=cp.OUT;AN=OUT/'analysis';AN.mkdir(exist_ok=True)
MODELS=['LI','SeasonalNaive','Spatial-Ridge','AG-LightGBM','AG-RandomForest','AG-DeepAR','AG-PatchTST','SAITS','SAITS-matched-local','SAITS-spatial','MOMENT','MOMENT-FT','Chronos2','TimesFM2.5','TimesFM3.0','TimesFM3.0-MV','TimesFM3.0-COV','TimesFM3.0-COV-SPA','BiTFI-TimesFM3','BiTFI-TimesFM3-fwd','BiTFI-Chronos2']

def save_merged(name,frames,parts=None):
 folder=OUT/'evaluation'/name;folder.mkdir(exist_ok=True);d=pd.concat(frames,ignore_index=True);d['model']=name;d['source_protocol']=d.get('source_protocol',d.get('protocol','unspecified'));d['protocol']=cp.PROTOCOL
 manifest=pd.read_csv(OUT/'mask_manifest.csv');ref={(int(r.case_id),v) for r in manifest.itertuples() for v in r.masked_vars.split(',')};q=d.query("group_type=='all'")
 assert set(zip(q.case_id,q.variable))==ref and not q.duplicated(['case_id','variable']).any()
 d.to_csv(folder/'results.csv',index=False)
 if parts:
  ids=np.concatenate([x['case_id'] for x in parts]);idx=ids.argsort();assert np.array_equal(ids[idx],manifest.case_id.to_numpy())
  np.savez_compressed(folder/'predictions.npz',case_id=ids[idx],prediction=np.concatenate([x['prediction'] for x in parts])[idx])
 (folder/'merged_complete.json').write_text(json.dumps(dict(cases=len(manifest),variable_cases=len(ref),source='This hourly run only'),indent=2))
 return d

def collect():
 depth=json.loads((OUT/'refinement_validation/selection.json').read_text())['selected_refinements']
 refinements=[];timing={i:0. for i in range(6)};timing_cases=0
 for k in [0,1,2,3,5]:
  paths=[OUT/f'revision/refinement/shard{i}' for i in range(3)];frames=[pd.read_csv(p/f'refine{k}.csv') for p in paths];d=pd.concat(frames,ignore_index=True);refinements.append(d.query("group_type=='all'"))
  if k==depth:save_merged('BiTFI-TimesFM3',frames,[np.load(p/f'refine{k}.npz') for p in paths])
 for i in range(3):
  info=json.loads((OUT/f'revision/refinement/shard{i}/complete.json').read_text())
  if 'A6000' in info['device']:
   timing_cases+=info['cases']
   for j,value in info['seconds_per_stage'].items():timing[int(j)]+=value
 cumulative={str(k):sum(timing[i] for i in range(k+1))/timing_cases for k in [0,1,2,3,5]}
 (AN/'a6000_timing.json').write_text(json.dumps(dict(cases=timing_cases,seconds_per_mask_by_refinements=cumulative,scope='Synchronized backbone calls, fusion and refinement covariate construction; excludes loading, initial mask construction and scoring',selected_refinements=depth),indent=2))
 for name,n in [('BiTFI-TimesFM3-fwd',2),('BiTFI-Chronos2',1)]:
  paths=[OUT/f'evaluation/{name}/shard{i}' for i in range(n)];save_merged(name,[pd.read_csv(p/'results.csv') for p in paths],[np.load(p/'predictions.npz') for p in paths])
 for name in ['SAITS-matched-local','SAITS-spatial']:
  path=OUT/'revision/saits_information/evaluation';save_merged(name,[pd.read_csv(path/f'{name}.csv')],[np.load(path/f'{name}.npz')])
 ref=pd.concat(refinements,ignore_index=True);ref.to_csv(AN/'refinement_cases.csv',index=False)
 summary(ref).to_csv(AN/'refinement_summary.csv',index=False);summary(ref,extra=['variable']).to_csv(AN/'refinement_variables.csv',index=False);site_scores(ref).to_csv(AN/'refinement_sites.csv',index=False)
 compare(site_scores(ref),[(f'BiTFI-refine{depth}',f'BiTFI-refine{k}') for k in [0,1,2,3,5] if k!=depth]).to_csv(AN/'refinement_paired.csv',index=False)
 frames=[];expected=None
 for m in MODELS:
  d=pd.read_csv(OUT/'evaluation'/m/'results.csv');assert set(d.model)=={m};d['source_protocol']=d.get('source_protocol',d.get('protocol','unspecified'));d['protocol']=cp.PROTOCOL;keys=set(map(tuple,d.query("group_type=='all'")[['case_id','variable']].to_numpy()))
  if expected is None:expected=keys
  assert keys==expected and np.isfinite(d.NMAE).all();frames.append(d)
 full=pd.concat(frames,ignore_index=True)
 provenance=[]
 for m,d in zip(MODELS,frames):
  path=OUT/'evaluation'/m/'results.csv';provenance.append(dict(model=m,result_file=str(path.relative_to(OUT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_protocols=sorted(d.source_protocol.dropna().unique().tolist())))
 (AN/'source_manifest.json').write_text(json.dumps(dict(run_root=str(OUT),sources=provenance,run_provenance_file='verification_20260929/provenance.json'),indent=2))
 full.to_csv(AN/'comparison_results.csv',index=False);base=full.query("group_type=='all'")
 for metric,extra,file in [('NMAE',[],'all_summary'),('NMAE',['variable'],'all_variables'),('MAE',['variable'],'all_physical_MAE'),('NMAE',['scenario','gap_length_h'],'all_gaps'),('NMAE',['scenario'],'all_scenarios')]:summary(base,metric,extra).to_csv(AN/(file+'.csv'),index=False)
 summary(full.query("group_type=='season'"),extra=['group_value']).to_csv(AN/'all_seasons.csv',index=False);site_scores(base).to_csv(AN/'all_site_scores.csv',index=False)
 compare(site_scores(base),[('BiTFI-TimesFM3',m) for m in MAIN_MODELS if m!='BiTFI-TimesFM3']).to_csv(AN/'main_paired.csv',index=False)
 compare(site_scores(base),[('BiTFI-TimesFM3','BiTFI-TimesFM3-fwd'),('BiTFI-TimesFM3','BiTFI-Chronos2'),('SAITS-spatial','SAITS-matched-local'),('MOMENT-FT','MOMENT'),('TimesFM3.0-COV','TimesFM3.0'),('TimesFM3.0-COV-SPA','TimesFM3.0-COV')]).to_csv(AN/'component_paired.csv',index=False)
 for family in [['SAITS','SAITS-matched-local','SAITS-spatial'],['MOMENT','MOMENT-FT'],['TimesFM3.0','TimesFM3.0-MV','TimesFM3.0-COV','TimesFM3.0-COV-SPA']]:
  means=summary(base[base.model.isin(family)]).set_index('model')['mean'];winner=means.idxmin();assert winner in MAIN_MODELS,(winner,'Update the representative figure set explicitly')
 additional=pd.read_csv(OUT/'additional_sites/results.csv');summary(additional).to_csv(AN/'additional_summary.csv',index=False);summary(additional,extra=['variable']).to_csv(AN/'additional_variables.csv',index=False);site_scores(additional).to_csv(AN/'additional_sites.csv',index=False);compare(site_scores(additional),[('BiTFI',m) for m in ['LI','Spatial-Ridge','TimesFM3-univariate']]).to_csv(AN/'additional_paired.csv',index=False)
 c=base.query("model=='BiTFI-TimesFM3' and scenario=='C' and variable in ['Tin','RH','CO2']").copy();c['model']='BiTFI-matched-C';b=pd.read_csv(OUT/'matched_availability/results.csv').query("group_type=='all'")
 assert set(zip(c.case_id,c.variable))==set(zip(b.case_id,b.variable))
 matched=pd.concat([b,c]);matched.to_csv(AN/'matched_availability_cases.csv',index=False);summary(matched).to_csv(AN/'matched_availability_summary.csv',index=False);summary(matched,extra=['variable']).to_csv(AN/'matched_availability_variables.csv',index=False);compare(site_scores(matched),[('BiTFI-matched-B','BiTFI-matched-C')]).to_csv(AN/'matched_availability_paired.csv',index=False)
 manifest=pd.read_csv(OUT/'mask_manifest.csv');q=base[base.model=='BiTFI-TimesFM3'].copy();total=q.groupby('greenhouse').n_eval.sum();q['weight']=q.n_eval/q.greenhouse.map(total)/len(total)
 weights=q.groupby(['scenario','gap_length_h']).agg(variable_cases=('variable','size'),sites=('greenhouse','nunique'),evaluated_hours=('n_eval','sum'),overall_weight=('weight','sum')).reset_index();counts=manifest.groupby(['scenario','gap_length_h']).size().rename('mask_cases').reset_index();weights.merge(counts).to_csv(AN/'scenario_duration_weights.csv',index=False)
 return full

def reporting_sensitivity(full=None):
 """Aggregate existing predictions; no model fitting or inference."""
 if full is None:full=pd.read_csv(AN/'comparison_results.csv')
 base=full.query("group_type=='all'")
 ref=pd.read_csv(AN/'refinement_cases.csv')
 summary(ref,extra=['scenario']).to_csv(AN/'refinement_scenarios.csv',index=False)
 site_scores(ref,extra=['scenario']).to_csv(AN/'refinement_scenario_sites.csv',index=False)
 # Every artificial masking job has one vote, independent of duration and sensor count.
 cases=base.groupby(['model','greenhouse','case_id'],as_index=False).NMAE.mean()
 cases['n_eval']=1
 summary(cases).to_csv(AN/'equal_case_summary.csv',index=False)
 site_scores(cases).to_csv(AN/'equal_case_sites.csv',index=False)
 compare(site_scores(cases),[('BiTFI-TimesFM3','TimesFM3.0-COV-SPA')]).to_csv(AN/'equal_case_paired.csv',index=False)
 uni=pd.concat([pd.read_csv(OUT/f'univariate_backbone_comparison/{m}_results.csv') for m in ['Chronos2','TimesFM2.5','TimesFM3.0']])
 compare(site_scores(uni),[('TimesFM3.0-UNI',m) for m in ['Chronos2-UNI','TimesFM2.5-UNI']]).to_csv(AN/'univariate_paired.csv',index=False)
 weights=pd.read_csv(AN/'scenario_duration_weights.csv').groupby('gap_length_h').overall_weight.sum()
 a=ref.query("scenario=='A' and refinements>=1").pivot(index=['case_id','variable'],columns='refinements',values='NMAE')
 assert (a.subtract(a[1],axis=0)==0).all().all()
 (AN/'reporting_sensitivity.json').write_text(json.dumps(dict(equal_case_definition='Average target-variable NMAEs within each mask, then masks equally within each greenhouse, then greenhouses equally',long_gap_weight=float(weights.loc[[72,168]].sum()),duration_weights=weights.to_dict(),scenario_A_refinement_1_to_5_identical_case_scores=True),indent=2))
 print('Scenario depth, equal-case sensitivity and all test families aggregated',flush=True)

def diagnostics(full):
 manifest=pd.read_csv(OUT/'mask_manifest.csv');sites={n:cp.site(n) for n in manifest.greenhouse.unique()};flags={n:np.load(OUT/'quality'/f'{n}_flags.npz')['constant'] for n in sites};flag_cases=[]
 for r in manifest.itertuples():
  for v in r.masked_vars.split(','):
   hours=int(flags[r.greenhouse][r.start_idx:r.end_idx+1,cp.COLS.index(v)].sum())
   flag_cases.append(dict(case_id=int(r.case_id),variable=v,constant_hours=hours))
 refs=pd.read_csv(OUT/'quality/reference_availability.csv');reference=full.query("group_type=='all' and model=='BiTFI-TimesFM3'").merge(refs[['case_id','variable','n_references']],on=['case_id','variable'],validate='one_to_one');summary(reference,extra=['n_references']).to_csv(AN/'reference_availability_summary.csv',index=False);reference.groupby(['variable','n_references']).agg(variable_cases=('case_id','size'),sites=('greenhouse','nunique'),evaluated_hours=('n_eval','sum')).reset_index().to_csv(AN/'reference_availability_counts.csv',index=False)
 flagged=pd.DataFrame(flag_cases);flagged.to_csv(AN/'constant_case_flags.csv',index=False);base=full.query("group_type=='all'").merge(flagged,on=['case_id','variable'],validate='many_to_one');summary(base[base.constant_hours==0]).to_csv(AN/'constant_exclusion_summary.csv',index=False)
 thresholds=json.loads((OUT/'quality/change_thresholds.json').read_text());records=[]
 for name in ['BiTFI-TimesFM3','TimesFM3.0-COV-SPA','SAITS-spatial','MOMENT-FT','Spatial-Ridge']:
  folder=OUT/'evaluation'/name;path=folder/'predictions.npz';preds=np.load(path)['prediction'] if path.exists() else None
  for r in manifest.itertuples():
   o=sites[r.greenhouse];pred=preds[r.case_id,:r.gap_length_h] if preds is not None else np.load(folder/'predictions'/f'{r.case_id}.npy');raw=o['data_raw'];sl=slice(r.start_idx,r.end_idx+1)
   for v in r.masked_vars.split(','):
    j=cp.COLS.index(v);truth=raw[v].iloc[sl].to_numpy();y=o['data'][v].iloc[sl].to_numpy();err=np.abs(pred[:,j]-y)*float(o['scaler'][v].data_range_[0]);delta=raw[v].diff().abs().iloc[sl].to_numpy();span=float(raw[v].max()-raw[v].min());groups={'change_low':np.isfinite(delta)&(delta<=thresholds[v]),'change_high':np.isfinite(delta)&(delta>thresholds[v])}
    if v=='Rad':groups.update(radiation_zero=truth==0,radiation_positive=truth>0)
    for group,mask in groups.items():
     if mask.any():records.append(dict(model=name,case_id=r.case_id,greenhouse=r.greenhouse,variable=v,stratum=group,n_eval=int(mask.sum()),MAE=float(err[mask].mean()),NMAE=float(err[mask].mean()/span)))
 d=pd.DataFrame(records);d.to_csv(AN/'residual_strata_cases.csv',index=False);summary(d,'MAE',['variable','stratum']).to_csv(AN/'residual_strata_MAE.csv',index=False);summary(d,extra=['variable','stratum']).to_csv(AN/'residual_strata_NMAE.csv',index=False)
 print('Analysis complete',AN,flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--reporting-only',action='store_true');args=parser.parse_args()
 if args.reporting_only:reporting_sensitivity()
 else:
  full=collect();diagnostics(full);reporting_sensitivity(full)

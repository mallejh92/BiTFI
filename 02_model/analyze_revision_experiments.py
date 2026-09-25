"""Reproducible, greenhouse-level analysis of prespecified revision experiments."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,spearmanr
import clean_protocol as cp
ROOT=cp.ROOT;RUN=ROOT/'03_result/revision_experiments_20260926';OUT=RUN/'analysis';OUT.mkdir(exist_ok=True)
MAIN=ROOT/'03_result/reevaluation_context_20260925/evaluation'

def site_scores(d,metric='NMAE',extra=()):
 keys=['model','greenhouse',*extra]
 z=d.assign(weighted=d[metric]*d.n_eval).groupby(keys,observed=True)[['weighted','n_eval']].sum()
 return (z.weighted/z.n_eval).rename(metric).reset_index()
def summary(d,metric='NMAE',extra=()):
 s=site_scores(d,metric,extra);rows=[]
 for key,g in s.groupby(['model',*extra],observed=True):
  if not isinstance(key,tuple):key=(key,)
  x=g[metric].to_numpy();i=np.random.default_rng(42).integers(len(x),size=(10000,len(x)));lo,hi=np.quantile(x[i].mean(1),[.025,.975])
  rows.append(dict(zip(['model',*extra],key))|dict(mean=x.mean(),sd=x.std(ddof=1),low=lo,high=hi,n_sites=len(x)))
 return pd.DataFrame(rows)
def compare(s,pairs,metric='NMAE'):
 p=s.pivot(index='greenhouse',columns='model',values=metric);rows=[]
 for a,b in pairs:
  z=p[[a,b]].dropna();x=z[a].to_numpy();y=z[b].to_numpy();i=np.random.default_rng(42).integers(len(x),size=(10000,len(x)));red=100*(1-x.mean()/y.mean());lo,hi=np.quantile(100*(1-x[i].mean(1)/y[i].mean(1)),[.025,.975]);pv=wilcoxon(x,y).pvalue if np.any(x!=y) else 1.
  rows.append(dict(model=a,reference=b,reduction=red,low=lo,high=hi,p=pv,n_sites=len(x)))
 d=pd.DataFrame(rows);ix=np.argsort(d.p.to_numpy());q=np.minimum(1,np.maximum.accumulate(d.p.to_numpy()[ix]*(len(d)-np.arange(len(d)))));d['holm_p']=0.;d.loc[ix,'holm_p']=q;return d

def collect():
 frames=[];paths={}
 for step in [0,1,2,3,5]:
  folder=RUN/'refinement';d=pd.concat([pd.read_csv(folder/f'shard{k}/refine{step}.csv') for k in range(3)],ignore_index=True);d=d.query("group_type=='all'");corrections=sorted((folder/'production_groups').glob('shard*/refine'+str(step)+'.csv'))
  if corrections:
   fix=pd.concat([pd.read_csv(p).query("group_type=='all'") for p in corrections]);d=pd.concat([d[~d.case_id.isin(fix.case_id)],fix],ignore_index=True)
  assert len(d)==6085 and d.case_id.nunique()==3357 and not d.duplicated(['case_id','variable']).any();frames.append(d)
  parts=[np.load(folder/f'shard{k}/refine{step}.npz') for k in range(3)];ids=np.concatenate([p['case_id'] for p in parts]);order=ids.argsort();assert np.array_equal(ids[order],np.arange(3357));a=np.concatenate([p['prediction'] for p in parts])[order]
  for p in sorted((folder/'production_groups').glob('shard*/refine'+str(step)+'.npz')):
   q=np.load(p);a[q['case_id']]=q['prediction']
  path=folder/f'refine{step}.npz';np.savez_compressed(path,case_id=ids[order],prediction=a);paths['BiTFI-refine'+str(step)]=path
 d=pd.concat(frames);d.to_csv(OUT/'refinement_cases.csv',index=False);s=site_scores(d);s.to_csv(OUT/'refinement_sites.csv',index=False);summary(d).to_csv(OUT/'refinement_summary.csv',index=False);summary(d,extra=['variable']).to_csv(OUT/'refinement_variables.csv',index=False)
 compare(s,[('BiTFI-refine1','BiTFI-refine'+str(i)) for i in [0,2,3,5]]).to_csv(OUT/'refinement_paired.csv',index=False)
 orig=pd.read_csv(MAIN/'DAFI-TimesFM3/results.csv').query("group_type=='all'");one=d.query('refinements==1');check=one.merge(orig,on=['case_id','variable'],suffixes=('_new','_main'),validate='one_to_one');delta=(check.NMAE_new-check.NMAE_main).abs();assert delta.max()<1.01e-5
 (OUT/'one_pass_equivalence.json').write_text(json.dumps(dict(cells=len(check),max_NMAE_difference=delta.max(),mean_NMAE_difference=delta.mean(),explanation='Same frozen weights/algorithm; batch-sensitive cases recomputed in original production groups; remaining numeric variation below 1.01e-5 case NMAE'),indent=2))
 sf=[]
 for name in ['SAITS-matched-local','SAITS-spatial']:
  folder=RUN/'saits_information/evaluation';z=pd.read_csv(folder/(name+'.csv')).query("group_type=='all'");assert len(z)==6085;sf.append(z);paths[name]=folder/(name+'.npz')
 saits=pd.concat(sf);summary(saits).to_csv(OUT/'saits_summary.csv',index=False);compare(site_scores(saits),[('SAITS-spatial','SAITS-matched-local')]).to_csv(OUT/'saits_paired.csv',index=False)
 add=pd.read_csv(RUN/'additional_sites/results.csv');assert add.greenhouse.nunique()==9;summary(add).to_csv(OUT/'additional_summary.csv',index=False);site_scores(add).to_csv(OUT/'additional_sites.csv',index=False);summary(add,extra=['variable']).to_csv(OUT/'additional_variables.csv',index=False);compare(site_scores(add),[('BiTFI',m) for m in ['LI','Spatial-Ridge','TimesFM3-univariate']]).to_csv(OUT/'additional_paired.csv',index=False)
 return paths

def agronomy(paths):
 manifest=pd.read_csv(cp.OUT/'mask_manifest.csv');sites={n:cp.site(n) for n in manifest.greenhouse.unique()};predictions={n:np.load(p)['prediction'] for n,p in paths.items() if n in ['BiTFI-refine0','BiTFI-refine1','SAITS-matched-local','SAITS-spatial']};derived=[];illum=[];outside=[]
 def vpd(a):return .6108*np.exp(17.27*a[:,0]/(a[:,0]+237.3))*(1-a[:,2]/100)
 for r in manifest.itertuples():
  o=sites[r.greenhouse];raw=o['data_raw'][cp.COLS].iloc[r.start_idx:r.end_idx+1].to_numpy(np.float64);rad=raw[:,4];valid_rad=np.isfinite(rad);base=dict(case_id=r.case_id,greenhouse=r.greenhouse,scenario=r.scenario,gap_length_h=r.gap_length_h)
  arrays={n:p[r.case_id,:r.gap_length_h].astype(np.float64) for n,p in predictions.items()}
  # Linear interpolation is deterministic and uses only masked observations.
  mr=cp.masked_case(o,r);arrays['LI']=mr.masked_data[cp.COLS].interpolate(limit_direction='both').ffill().bfill().iloc[r.start_idx:r.end_idx+1].to_numpy(np.float64)
  for name,a in arrays.items():
   for j,v in enumerate(cp.COLS):a[:,j]=a[:,j]*o['scaler'][v].data_range_[0]+o['scaler'][v].data_min_[0]
   for j,v in enumerate(cp.COLS):
    if v not in r.masked_vars.split(','):continue
    good=np.isfinite(raw[:,j]);assert np.isfinite(a[good,j]).all()
    for period,mask in [('light',rad>20),('low-light',rad<=20)]:
     keep=good&valid_rad&mask
     if keep.sum()>=2:illum.append(base|dict(model=name,variable=v,period=period,MAE=np.abs(a[keep,j]-raw[keep,j]).mean(),n_eval=int(keep.sum())))
   if r.scenario in ['B','C']:
    keep=np.isfinite(raw[:,[0,2]]).all(1);assert keep.all();truth=vpd(raw);estimated=vpd(a);assert np.isfinite(estimated).all();derived.append(base|dict(model=name,metric='VPD',MAE=np.abs(estimated-truth).mean(),bias=(estimated-truth).mean(),n_eval=len(raw)))
    outside.append(base|dict(model=name,negative_vpd=int((estimated<0).sum()),n_eval=len(raw)))
   if 'Rad' in r.masked_vars.split(','):
    assert valid_rad.all();err=(a[:,4].sum()-rad.sum())*.0036;derived.append(base|dict(model=name,metric='radiation_integral',MAE=abs(err),bias=err,n_eval=1))
 d=pd.DataFrame(derived);d.to_csv(OUT/'agronomic_cases.csv',index=False);summary(d,'MAE',['metric']).to_csv(OUT/'agronomic_summary.csv',index=False);summary(d,'MAE',['metric','gap_length_h']).to_csv(OUT/'agronomic_by_gap.csv',index=False);site_scores(d,'MAE',['metric']).to_csv(OUT/'agronomic_sites.csv',index=False)
 i=pd.DataFrame(illum);i.to_csv(OUT/'illumination_cases.csv',index=False);summary(i,'MAE',['variable','period']).to_csv(OUT/'illumination_summary.csv',index=False);pd.DataFrame(outside).to_csv(OUT/'vpd_physical_checks.csv',index=False)
 base=pd.concat([pd.read_csv(MAIN/m/'results.csv').query("group_type=='all'") for m in ['DAFI-TimesFM3','Spatial-Ridge','SAITS','MOMENT-FT','LI']]);summary(base,'MAE',['variable']).to_csv(OUT/'physical_MAE.csv',index=False)
 b=base.query("model=='DAFI-TimesFM3' and variable in ['Tin','RH','CO2']");summary(b,extra=['scenario']).to_csv(OUT/'indoor_scenarios.csv',index=False);summary(b,extra=['scenario','gap_length_h']).to_csv(OUT/'indoor_scenarios_gap.csv',index=False)
 common=set(b.query("scenario=='B'").greenhouse)&set(b.query("scenario=='C'").greenhouse);bc=b[b.greenhouse.isin(common)&b.scenario.isin(['B','C'])];summary(bc,extra=['scenario']).to_csv(OUT/'indoor_scenarios_common_sites.csv',index=False)
 variability=[];ss=site_scores(base.query("model=='DAFI-TimesFM3'"),extra=['variable'])
 for n,o in sites.items():
  for v in cp.COLS:
   series=o['data_raw'][v];adjacent=series.diff().abs();span=series.max()-series.min();variability.append(dict(greenhouse=n,variable=v,median_hourly_change=adjacent.median(),mean_hourly_change=adjacent.mean(),scaled_hourly_change=adjacent.mean()/span,range=span))
 vv=pd.DataFrame(variability).merge(ss,on=['greenhouse','variable']);vv.to_csv(OUT/'variability_sites.csv',index=False);corr=[]
 for v,g in vv.groupby('variable'):
  r,p=spearmanr(g.scaled_hourly_change,g.NMAE);corr.append(dict(variable=v,rho=r,p=p,n_sites=len(g)))
 pd.DataFrame(corr).to_csv(OUT/'variability_correlations.csv',index=False)
 (OUT/'agronomic_protocol.json').write_text(json.dumps(dict(air_vpd='0.6108 exp(17.27 T/(T+237.3)) (1-RH/100), hourly Tin/RH, scenarios B/C only; no clipping',radiation='0.0036 times summed hourly W/m2, MJ/m2 per gap; each gap equally weighted within site',illumination='Observed outdoor radiation >20 versus <=20 W/m2; diagnostic grouping only, never model input when hidden',aggregation='Hours within site for pointwise metrics; gaps within site for radiation integral; sites equally weighted; 10000 bootstrap resamples seed42',interpretation='Post hoc agronomic diagnostics, not crop outcome validation; hourly radiation assumed representative of hourly interval'),indent=2))
if __name__=='__main__':
 paths=collect();agronomy(paths);print('Analysis complete',OUT)

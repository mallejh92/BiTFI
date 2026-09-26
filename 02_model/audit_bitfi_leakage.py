"""Read-only audit of the existing BiTFI evaluation. Never rewrites benchmark results."""
from pathlib import Path
import argparse,contextlib,io,json,zlib,hashlib
import numpy as np,pandas as pd
from sklearn.preprocessing import MinMaxScaler
from bitfi import BiTFIImputation,BiTFITimesFM3
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'03_result/leakage_audit_20260911';OUT.mkdir(exist_ok=True)
def save(name,obj):
 (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2));print(name,json.dumps(obj),flush=True)
class Probe(BiTFIImputation):
 def _ready(self):return True
 def _max_h(self):return 2048
 def _fc_uni(self,ctx,h):return np.full(h,np.mean(ctx),np.float32)
 def _fc_cov(self,ctx,names,pf,h):return np.full(h,np.mean(ctx)+.1*np.mean(pf),np.float32)
def core(real=False):
 n=600;index=pd.date_range('2025-01-01',periods=n,freq='h');t=np.arange(n)
 x=pd.DataFrame({c:.4+.1*np.sin(t/9+i) for i,c in enumerate(['Tin','Tout','RH','CO2','Rad'])},index=index)
 x.loc[index[200:210],'Tin']=np.nan
 art=pd.DataFrame(False,index=index,columns=x.columns);art.iloc[250:322,:]=True
 changed=x.copy();changed[art]=.9
 masked=x.mask(art);valid=x.notna().astype('float32');effective=valid.mask(art,0)
 model=BiTFITimesFM3(context_len=1440,use_spatial=False) if real else Probe(context_len=1440,use_spatial=False)
 assert model._ready(), 'Backend unavailable: refuse fallback'
 b0=model.compute_base(x,valid);b1=model.compute_base(changed,valid)
 model.set_greenhouse('audit');p0=model.impute_artificial(masked,effective,art,b0)
 poisoned=b0.copy();poisoned[art]=1000000
 model.set_greenhouse('audit');pp=model.impute_artificial(masked,effective,art,poisoned)
 model.set_greenhouse('audit');p1=model.impute_artificial(masked,effective,art,b1)
 save('real_backend' if real else 'core_probe',dict(input='Synthetic normalized 600-hour five-variable series, natural Tin gap 200:210, artificial all-variable gap 250:322; no neighbors',actual_timesfm3=real,hidden_base_values_only_max_prediction_change=float(np.max(np.abs((p0-pp).values[art.values]))),hidden_truth_changed_before_compute_base_max_prediction_change=float(np.max(np.abs((p0-p1).values[art.values]))),natural_gap_cached_estimate_change=float(np.max(np.abs((b0-b1).iloc[200:210,0])))))
 if real:return
 class NotReady(Probe):
  def _ready(self):return False
 fallback=NotReady().impute_artificial(masked,effective,art,b0)
 save('backend_failure',dict(returned_hidden_truth_exactly=bool(np.array_equal(fallback.values[art.values],b0.values[art.values]))))
 class RecordingBank:
  def __init__(self):self.calls=[]
  def select(self,col,index,target,observed,k):
   self.calls.append(observed.copy());return np.zeros((1,len(index)),np.float32),['synthetic_neighbor']
 bank=RecordingBank();m=Probe(bank=bank);m.set_greenhouse('same_site');obs1=np.ones(n,bool);obs1[250:256]=False;obs2=np.ones(n,bool);obs2[300:372]=False
 m._neighbors('Tin',index,x.Tin.values,obs1);m._neighbors('Tin',index,x.Tin.values,obs2)
 save('cache_scope',dict(neighbor_select_calls=len(bank.calls),expected_independent_mask_calls=2,later_hidden_hours_visible_at_cached_selection=int((bank.calls[0]&~obs2).sum())))
 from preprocessing import _interpolate_short_gaps
 q=pd.DataFrame({'Tin':[10.,np.nan,20.,30.]});r=q.copy();r.iloc[2,0]=60
 a=_interpolate_short_gaps(q);b=_interpolate_short_gaps(r)
 save('preprocessing_probe',dict(hidden_position=2,visible_interpolated_position=1,before=float(a.iloc[1,0]),after=float(b.iloc[1,0]),full_series_scaler_max_before=float(MinMaxScaler().fit(a).data_max_[0]),full_series_scaler_max_after=float(MinMaxScaler().fit(b).data_max_[0])))
def data_audit():
 import preprocessing as pre
 from masking_v2 import create_gap_masks,SCENARIO_CONFIGS
 split=json.loads((ROOT/'03_result/comparison_bitfi_tfm3/split.json').read_text());train={Path(p).resolve() for p in split['train']};test={Path(p).resolve() for p in split['test']}
 hashes=lambda paths:{hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
 save('split',dict(train_test_path_overlap=[str(p) for p in train&test],identical_train_test_file_hashes=list(hashes(train)&hashes(test)),train_candidates=len(train),test_sites=len(test)))
 orig=pre._interpolate_short_gaps;capture={}
 def record(df,limit=pre.SHORT_GAP_LIMIT):capture['pre']=df.copy();return orig(df,limit)
 pre._interpolate_short_gaps=record;rows=[];cols=list(pre.TARGET_COLS.values());selection=json.loads((ROOT/'03_result/figure6_common_window/selection.json').read_text());figure=[]
 for fp in split['test']:
  with contextlib.redirect_stdout(io.StringIO()):res=pre.preprocess_file(fp,include_covariates=True)
  df=res['data'][cols];raw=capture['pre'][cols];valid=df.notna().astype('float32');name=res['name']
  for sc,combos in SCENARIO_CONFIGS.items():
   for vs in combos:
    for h in [6,12,24,72,168]:
     seed=42+zlib.crc32(f'{name}{sc}{h}'.encode())%9999
     for mr in create_gap_masks(df,valid,sc,h,vs,context_len_h=1440,n_repeats=10,random_seed=seed):
      for c in vs:
       sel=mr.eval_mask[c];outside=df.loc[~sel,c];fullrange=float(df[c].max()-df[c].min());obsrange=float(outside.max()-outside.min())
       rows.append(dict(greenhouse=name,scenario=sc,variable=c,gap_h=h,repeat=mr.repeat,start=str(df.index[mr.gap_start_idx]),end=str(df.index[mr.gap_end_idx]),n_eval=int(sel.sum()),not_raw_observed_eval=int((sel&raw[c].isna()).sum()),scaler_extrema_change=bool(fullrange!=obsrange)))
  if name==selection['greenhouse']:
   ix=(df.index>=selection['start'])&(df.index<=selection['end'])
   for c in cols:figure.append(dict(variable=c,gap_hours=int(ix.sum()),not_raw_observed_truth=int(raw.loc[ix,c].isna().sum())))
  print('audited',name,flush=True)
 d=pd.DataFrame(rows);d.to_csv(OUT/'evaluation_truth_audit.csv',index=False)
 save('evaluation_truth_summary',dict(variable_cases=len(d),evaluated_point_occurrences=int(d.n_eval.sum()),not_raw_observed_point_occurrences=int(d.not_raw_observed_eval.sum()),cases_containing_interpolated_truth=int(d.not_raw_observed_eval.gt(0).sum()),cases_with_mask_dependent_scaler_extrema=int(d.scaler_extrema_change.sum()),figure6_truth=figure,counts_note='Repeated evaluation occurrences, not unique timestamps; raw means after physical-range filtering but before interpolation'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['core','real','data']);a=p.parse_args()
 if a.mode=='data':data_audit()
 else:core(a.mode=='real')

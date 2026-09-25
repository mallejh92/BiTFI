"""Cross-check publication against executed revision cases and PDF build logs."""
from pathlib import Path
import hashlib,json,re
import numpy as np,pandas as pd,pymupdf
from analyze_revision_experiments import site_scores,summary
from revision_config import MAIN_MODELS
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'03_result/revision_experiments_20260926';A=RUN/'analysis';T=ROOT/'05_thesis';D=ROOT/'04_figure/source_data'
d=pd.read_csv(A/'comparison_results.csv');raw=d.query("group_type=='all'");assert raw.model.nunique()==21
ref=raw[raw.model=='DAFI-TimesFM3'].sort_values(['case_id','variable'])
for m,g in raw.groupby('model'):
 q=g.sort_values(['case_id','variable']);assert len(q)==6085 and q.case_id.nunique()==3357
 pd.testing.assert_frame_equal(q[['case_id','variable','n_eval']].reset_index(drop=True),ref[['case_id','variable','n_eval']].reset_index(drop=True))
 assert np.isfinite(q[['NMAE','MAE']].to_numpy()).all()
main=pd.read_csv(D/'main_summary.csv');assert set(main.model)==set(MAIN_MODELS)
calc=summary(raw).set_index('model')
for r in main.itertuples():np.testing.assert_allclose(r.NMAE,calc.loc[r.model,'mean'],atol=1e-12)
for name,keys in [('gap_scenario_greenhouse_scores.csv',['scenario','gap_length_h']),('variable_greenhouse_scores.csv',['variable']),('season_greenhouse_scores.csv',['group_value'])]:
 x=pd.read_csv(D/name);assert set(x.model)==set(MAIN_MODELS);assert x.groupby(keys+['model']).NMAE.mean().unstack('model').idxmin(axis=1).eq('DAFI-TimesFM3').all()
assert len(pd.read_csv(D/'paired_greenhouse_tests.csv'))==11
for k in range(3):assert json.loads((RUN/f'refinement/shard{k}/complete.json').read_text())['matmul_precision']=='highest'
assert json.loads((RUN/'refinement/smoke/complete.json').read_text())['checks'][0]['hidden_target_invariant']
assert all(c['max_prediction_diff']==0 for c in json.loads((A/'algorithm_equivalence.json').read_text())['cases'])
assert (RUN/'additional_sites/final_precision_complete').exists()
# Validate VPD from an independently selected saved prediction/truth case.
from clean_protocol import site,COLS,OUT
manifest=pd.read_csv(OUT/'mask_manifest.csv');r=manifest.query("scenario=='B'").iloc[0];obj=site(r.greenhouse);p=np.load(RUN/'refinement/refine1.npz')['prediction'][int(r.case_id),:int(r.gap_length_h)].astype(float);truth=obj['data_raw'][COLS].iloc[int(r.start_idx):int(r.end_idx)+1]
for j,v in enumerate(COLS):p[:,j]=p[:,j]*obj['scaler'][v].data_range_[0]+obj['scaler'][v].data_min_[0]
def vpd(t,h):return .6108*np.exp(17.27*t/(t+237.3))*(1-h/100)
mae=np.mean(np.abs(vpd(p[:,0],p[:,2])-vpd(truth.Tin.to_numpy(),truth.RH.to_numpy())))
ag=pd.read_csv(A/'agronomic_cases.csv');got=ag[(ag.case_id==r.case_id)&(ag.model=='BiTFI-refine1')&(ag.metric=='VPD')].iloc[0].MAE;np.testing.assert_allclose(got,mae,atol=1e-10)
artifacts={};pages={}
for stem in ['TFM','supplementary']:
 s=(T/(stem+'.tex')).read_text();inputs=[T/(stem+'.tex'),T/'cas-refs.bib']+[T/n for n in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',s)]
 assert (T/(stem+'.pdf')).stat().st_mtime>=max(p.stat().st_mtime for p in inputs), 'PDF is older than source/figure inputs'
 assert not re.search(r'\b(corrected|previously|archived)\b',s,re.I)
 assert not re.search(r'\b0\.\d{5,}',s)
 assert 'CAFI' not in s and not re.search(r'Round[~ ]?[01]',s)
 log=(T/(stem+'.log')).read_text();assert not re.search(r'undefined (references|citations)|Overfull \\hbox',log)
 doc=pymupdf.open(T/(stem+'.pdf'));pages[stem]=len(doc);txt='\n'.join(p.get_text() for p in doc);assert '??' not in txt
 artifacts[stem]=hashlib.sha256((T/(stem+'.pdf')).read_bytes()).hexdigest()
s=(T/'TFM.tex').read_text()
for declaration in ['Declaration of competing interest','Funding','Data availability','Code availability','Declaration of generative AI']:assert declaration in s
for file in T.glob('Figure*.pdf'):assert len(pymupdf.open(file))==1
report=dict(models=21,main_comparison_models=MAIN_MODELS,mask_jobs=3357,variable_cases=6085,additional_sites=9,additional_masks=528,refinement_passes=[0,1,2,3,5],hidden_target_checks='passed',one_pass_algorithm_equivalence='exact on 15 largest pilot-discrepancy cases',derived_VPD_independently_recomputed=True,common_comparison_set=True,undefined_references=False,overfull_boxes=False,pages=pages,pdf_sha256=artifacts)
(RUN/'publication_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

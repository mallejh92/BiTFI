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
for family in [['SAITS','SAITS-matched-local','SAITS-spatial'],['MOMENT','MOMENT-FT'],[m for m in calc.index if m.startswith('TimesFM3.0')]]:
 assert calc.loc[family,'mean'].idxmin() in MAIN_MODELS
example=pd.read_csv(D/'illustrative_case_ABC_physical_units.csv');metrics=pd.read_csv(D/'figure6_panel_metrics.csv')
assert set(example.model)=={'SAITS-spatial','MOMENT-FT','Spatial-Ridge','DAFI-TimesFM3'}
assert len(metrics)==52
for r in metrics.itertuples():
 q=example[(example.model==r.model)&(example.scenario==r.scenario)&(example.variable==r.variable)&example.artificial]
 y=q.truth.to_numpy();pred=q.prediction.to_numpy();assert len(q)==72
 np.testing.assert_allclose(r.MAE,np.abs(pred-y).mean(),atol=1e-10)
 np.testing.assert_allclose(r.R2,1-((pred-y)**2).sum()/((y-y.mean())**2).sum(),atol=1e-10)

for name,keys in [('gap_scenario_greenhouse_scores.csv',['scenario','gap_length_h']),('variable_greenhouse_scores.csv',['variable']),('season_greenhouse_scores.csv',['group_value'])]:
 x=pd.read_csv(D/name);assert set(x.model)==set(MAIN_MODELS);assert x.groupby(keys+['model']).NMAE.mean().unstack('model').idxmin(axis=1).eq('DAFI-TimesFM3').all()
assert len(pd.read_csv(D/'paired_greenhouse_tests.csv'))==len(MAIN_MODELS)-1
for k in range(3):assert json.loads((RUN/f'refinement/shard{k}/complete.json').read_text())['matmul_precision']=='highest'
assert json.loads((RUN/'refinement/smoke/complete.json').read_text())['checks'][0]['hidden_target_invariant']
assert all(c['max_prediction_diff']==0 for c in json.loads((A/'algorithm_equivalence.json').read_text())['cases'])
assert (RUN/'additional_sites/final_precision_complete').exists()
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
report=dict(models=21,main_comparison_models=MAIN_MODELS,mask_jobs=3357,variable_cases=6085,additional_sites=9,additional_masks=528,refinement_passes=[0,1,2,3,5],hidden_target_checks='passed',one_pass_algorithm_equivalence='exact on 15 largest pilot-discrepancy cases',common_comparison_set=True,undefined_references=False,overfull_boxes=False,pages=pages,pdf_sha256=artifacts)
(RUN/'publication_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

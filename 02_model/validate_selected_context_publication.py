"""Audit publication statistics against case-level files before replacing outputs."""
from pathlib import Path
import json, hashlib, re
import numpy as np
import pandas as pd
import pymupdf
from sklearn.metrics import mean_absolute_error, r2_score
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'03_result/reevaluation_context_20260925';SRC=RUN/'figures/source_data';MAN=RUN/'manuscript'
assert (RUN/'validation_report.json').exists()
assert (RUN/'univariate_backbone_comparison/validation_report.json').exists()
summary=pd.read_csv(SRC/'main_summary.csv').set_index('model')
means={}
for p in (RUN/'evaluation').glob('*/complete.json'):
 if p.parent.name.startswith('CAFI') or 'pilot' in p.parent.name:continue
 d=pd.read_csv(p.parent/'results.csv').query("group_type=='all'")
 g=d.assign(weighted=d.NMAE*d.n_eval).groupby('greenhouse')[['weighted','n_eval']].sum()
 means[p.parent.name]=float((g.weighted/g.n_eval).mean())
 if p.parent.name in summary.index:np.testing.assert_allclose(means[p.parent.name],summary.loc[p.parent.name,'NMAE'],atol=1e-12)
assert len(means)==19 and min(means,key=means.get)=='BiTFI-TimesFM3'
for filename,levels in [('gap_scenario_greenhouse_scores.csv',['scenario','gap_length_h']),('variable_greenhouse_scores.csv',['variable']),('season_greenhouse_scores.csv',['group_value'])]:
 d=pd.read_csv(SRC/filename);g=d.groupby(levels+['model']).NMAE.mean().unstack('model')
 assert (g.idxmin(axis=1)=='BiTFI-TimesFM3').all(),filename
# All annotated examples use the actual newly exported predictions.
d=pd.read_csv(SRC/'illustrative_case_ABC_physical_units.csv');metrics=pd.read_csv(SRC/'figure6_panel_metrics.csv')
assert len(metrics)==52
for r in metrics.itertuples():
 q=d[(d.model==r.model)&(d.variable==r.variable)&(d.scenario==r.scenario)&d.artificial.astype(bool)]
 assert len(q)==72
 np.testing.assert_allclose([r.MAE,r.R2],[mean_absolute_error(q.truth,q.prediction),r2_score(q.truth,q.prediction)],rtol=1e-10,atol=1e-10)
nums=json.loads((RUN/'manuscript_assets/manuscript_assets.json').read_text())['generated_numbers']
for name in ['TFM','supplementary']:
 s=(MAN/(name+'.tex')).read_text()
 for key,value in nums.items():assert '\\newcommand{\\'+key+'}{'+value+'}' in s,(name,key)
 assert '13.46' not in s and 'Context lengths were set in earlier exploratory' not in s
 assert '1,900' in s and 'CAFI' not in s
 assert not re.search(r'\b0\.\d{5,}',s)
 log=(MAN/(name+'.log')).read_text(errors='replace')
 assert 'undefined references' not in log and 'undefined citations' not in log
 assert 'Overfull \\hbox' not in log
 doc=pymupdf.open(MAN/(name+'.pdf'));assert len(doc)>0
figures=list((RUN/'figures').glob('Figure*.pdf'))+list((RUN/'figures/Supplementary').glob('Figure*.pdf'))
assert len(figures)==14
for p in figures:assert len(pymupdf.open(p))==1
with pymupdf.open(RUN/'figures/Figure0_Graphical_abstract.pdf') as doc:
 text=doc[0].get_text().replace('\u00a0',' ')
 assert nums['CovReduction']+'% lower NMAE' in text
 assert 'vs. TimesFM3' in text and text.splitlines().count('Time')==2
report=dict(protocol='validation-context-20260925',model_means_verified=means,lowest_in_15_gap_conditions_5_variables_4_seasons=True,example_metrics_independently_recomputed=52,numerical_macros_verified=nums,figure_pdfs=14,latex_builds=['TFM','supplementary'],undefined_references=False,overfull_hboxes=False,artifacts={str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in figures+[MAN/'TFM.tex',MAN/'TFM.pdf',MAN/'supplementary.tex',MAN/'supplementary.pdf']})
(RUN/'publication_validation.json').write_text(json.dumps(report,indent=2));print('Publication numerical and build checks passed.')

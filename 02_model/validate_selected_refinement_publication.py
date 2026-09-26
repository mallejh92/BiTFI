"""Check that validation selection and published statistics refer to one setting."""
from pathlib import Path
import json,re,hashlib
import numpy as np,pandas as pd
import clean_protocol as cp
from analyze_revision_experiments import site_scores,summary
ROOT=cp.ROOT;V=ROOT/'03_result/refinement_validation_20260926';T=ROOT/'05_thesis';A=ROOT/'03_result/revision_experiments_20260926/analysis'
def main():
 sel=json.loads((V/'selection.json').read_text());k=sel['selected_refinements'];d=pd.read_csv(V/'results.csv');v=summary(d).set_index('model');expected=min([1,2,3,5],key=lambda i:(v.loc[f'BiTFI-refine{i}','mean'],i));assert k==expected
 published=pd.read_csv(V/'summary.csv').set_index('model')
 np.testing.assert_allclose(v.loc[published.index,'mean'],published['mean'],rtol=0,atol=1e-12)
 manifest=pd.read_csv(V/'mask_manifest.csv');assert len(manifest)==1898 and manifest.greenhouse.nunique()==21
 sites=pd.read_csv(cp.OUT/'sites.csv');assert set(manifest.greenhouse)<=set(sites.query("group=='train'").name);assert (manifest.start_idx>=manifest.validation_start).all()
 for name,g in manifest.groupby('greenhouse'):
  o=cp.site(name)
  for r in g.itertuples():assert o['data_raw'].iloc[r.start_idx:r.end_idx+1][r.masked_vars.split(',')].notna().all().all()
 for p in V.glob('shard*/complete.json'):
  x=json.loads(p.read_text());assert x['device']=='NVIDIA RTX A6000' and x['precision']=='highest';assert all(c['hidden_target_invariant'] and c['self_reference_excluded'] for c in x['checks'])
 active=ROOT/json.loads((ROOT/'03_result/active_evaluation.json').read_text())['result_root'];raw=pd.read_csv(A/'comparison_results.csv').query("group_type=='all'");ref=pd.read_csv(A/'refinement_cases.csv').query('refinements==@k');actual=raw.query("model=='DAFI-TimesFM3'");z=actual.merge(ref,on=['case_id','variable'],suffixes=('_main','_candidate'),validate='one_to_one');assert len(z)==6085
 np.testing.assert_allclose(z.NMAE_main,z.NMAE_candidate,rtol=0,atol=1.1e-5)
 means=summary(raw).set_index('model');s=(T/'TFM.tex').read_text()
 def macro(name):return re.search(r'\\newcommand\{\\'+name+r'\}\{([^}]+)\}',s).group(1)
 assert macro('BiTFINMAE')==f"{means.loc['DAFI-TimesFM3','mean']:.4f}"
 assert macro('ForwardNMAE')==f"{means.loc['DAFI-TimesFM3-fwd','mean']:.4f}"
 assert macro('ChronosBiTFI')==f"{means.loc['DAFI-Chronos2','mean']:.4f}"
 if k>1:
  for name in ['DAFI-TimesFM3','DAFI-TimesFM3-fwd','DAFI-Chronos2']:
   checks=json.loads((active/'smoke'/name/'shard0/complete.json').read_text())['checks'];assert all(c['depth']==k and c['hidden_target_invariant'] and c['execution_order_invariant'] for c in checks)
  x=json.loads((active/'additional_sites/complete.json').read_text());assert x['selected_refinements']==k and all(c['hidden_target_invariant'] for c in x['checks'])
 assert 'saved site split' not in s and 'computational compromise' not in s
 assert 'NVIDIA RTX A6000' in s
 assert not re.search(r'(?:one|single) (?:synchronous )?covariate-refinement pass',s) or k==1
 assert s.index('The forward-only BiTFI setting obtained')<s.index(r'\subsection{Information sources and backbone choice}')
 report=dict(selected_refinements=k,selection_recomputed=True,validation_sites=21,validation_mask_jobs=1898,validation_variable_cases=3332,test_sites_used_for_selection=0,selected_test_cases=len(z),control_hidden_target_checks=True,A6000_measurement_scope_documented=True,fig6_metrics_checked_by='validate_revision_publication.py',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [V/'protocol.json',V/'mask_manifest.csv',V/'selection.json',V/'summary.csv',A/'comparison_results.csv',T/'TFM.tex',T/'supplementary.tex']})
 (V/'publication_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

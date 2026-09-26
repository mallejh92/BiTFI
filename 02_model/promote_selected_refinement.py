"""Assemble executed validation-selected outputs without altering earlier runs."""
from pathlib import Path
import argparse,json,shutil,hashlib
import numpy as np,pandas as pd
import clean_protocol as cp
from analyze_revision_experiments import summary
ROOT=cp.ROOT;BASE=ROOT/'03_result/reevaluation_context_20260925';OUT=ROOT/'03_result/reevaluation_refinement_20260926';REV=ROOT/'03_result/revision_experiments_20260926'
def main():
 p=argparse.ArgumentParser();p.add_argument('--activate',action='store_true');a=p.parse_args();sel=json.loads((ROOT/'03_result/refinement_validation_20260926/selection.json').read_text());k=sel['selected_refinements'];OUT.mkdir(exist_ok=True)
 if k==1:print('Single-pass setting retained; no test-result substitution needed');return
 for name in ['data','models','univariate_backbone_comparison']:
  if not (OUT/name).exists():(OUT/name).symlink_to(BASE/name,target_is_directory=True)
 for name in ['sites.csv','mask_manifest.csv','split.json','scalers.pkl']:shutil.copy2(BASE/name,OUT/name)
 proto=json.loads((BASE/'protocol.json').read_text());proto.update(version='validation-refinement-20260926',selected_refinements=k,refinement_selection_source='03_result/refinement_validation_20260926/selection.json',refinement_selection=sel['selection']);(OUT/'protocol.json').write_text(json.dumps(proto,indent=2))
 (OUT/'evaluation').mkdir(exist_ok=True)
 for source in (BASE/'evaluation').iterdir():
  if source.name in ['BiTFI-TimesFM3','BiTFI-TimesFM3-fwd','BiTFI-Chronos2']:continue
  dest=OUT/'evaluation'/source.name
  if not dest.exists():dest.symlink_to(source,target_is_directory=True)
 folder=OUT/'evaluation/BiTFI-TimesFM3';folder.mkdir(exist_ok=True)
 parts=[REV/f'refinement/shard{i}/refine{k}.csv' for i in range(3)];d=pd.concat([pd.read_csv(p) for p in parts],ignore_index=True)
 fixes=sorted((REV/'refinement/production_groups').glob(f'shard*/refine{k}.csv'))
 if fixes:
  f=pd.concat([pd.read_csv(p) for p in fixes],ignore_index=True);d=pd.concat([d[~d.case_id.isin(f.case_id)],f],ignore_index=True)
 d['model']='BiTFI-TimesFM3';d=d.sort_values(['case_id','group_type','group_value','variable']);d.to_csv(folder/'results.csv',index=False)
 shutil.copy2(REV/f'refinement/refine{k}.npz',folder/'predictions.npz')
 info=summary(d.query("group_type=='all'")).iloc[0].to_dict();info.update(selected_refinements=k,mask_jobs=d.case_id.nunique(),source='Executed full-manifest refinement candidate, selected using training-site validation only',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*parts,*fixes]});(folder/'complete.json').write_text(json.dumps(info,indent=2))
 ef=OUT/'figure6_common_window';ef.mkdir(exist_ok=True)
 for p in (BASE/'figure6_common_window').iterdir():
  if p.is_file() and p.name!='BiTFI.csv':shutil.copy2(p,ef/p.name)
 if not a.activate:print('Prepared',OUT,'depth',k);return
 for name in ['BiTFI-TimesFM3-fwd','BiTFI-Chronos2']:
  folder=OUT/'evaluation'/name;parts=sorted(folder.glob(f'shard*/refine{k}.csv'))
  if not parts:parts=sorted(folder.glob('shard*/results.csv'))
  d=pd.concat([pd.read_csv(p) for p in parts]);assert d.refinements.eq(k).all();assert d.query("group_type=='all'").case_id.nunique()==3357
  d.to_csv(folder/'results.csv',index=False);z=summary(d.query("group_type=='all'")).iloc[0].to_dict();z.update(selected_refinements=k,mask_jobs=3357,shards=[json.loads(p.read_text()) for p in folder.glob('shard*/complete.json')]);(folder/'complete.json').write_text(json.dumps(z,indent=2))
 assert (OUT/'additional_sites/complete.json').exists() and (ef/'BiTFI.csv').exists()
 active=json.loads((ROOT/'03_result/active_evaluation.json').read_text());active.update(version='validation-refinement-20260926',result_root=str(OUT.relative_to(ROOT)),refinement_selection='03_result/refinement_validation_20260926',selected_refinements=k,models=21,supplementary_figures=9)
 (ROOT/'03_result/active_evaluation.json').write_text(json.dumps(active,indent=2));print('Activated selected depth',k)
if __name__=='__main__':main()

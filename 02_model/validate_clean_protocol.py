"""Independent invariants for the persisted evaluation inputs and completed results."""
import json,pickle,hashlib
import numpy as np,pandas as pd
import clean_protocol as cp

def main():
 table=pd.read_csv(cp.OUT/'sites.csv');sites={r.name:cp.site(r.name) for r in table.itertuples()}
 train={n:o for n,o in sites.items() if o['group']=='train'};test={n:o for n,o in sites.items() if o['group']=='test'}
 assert not set(train)&set(test)
 with (cp.OUT/'scalers.pkl').open('rb') as f:scalers=pickle.load(f)
 for c,scaler in scalers.items():
  values=np.concatenate([o['data_raw'][c].iloc[:o['cut']].dropna().values for o in train.values() if c in o['data_raw']])
  np.testing.assert_array_equal(scaler.data_min_,[values.min()]);np.testing.assert_array_equal(scaler.data_max_,[values.max()])
  assert scaler.n_samples_seen_==len(values)
 manifest=pd.read_csv(cp.OUT/'mask_manifest.csv');total=0
 for r in manifest.itertuples():
  obj=test[r.greenhouse];block=obj['data_raw'].iloc[r.start_idx:r.end_idx+1][r.masked_vars.split(',')]
  assert block.notna().all().all() and len(block)==r.gap_length_h
  assert r.start_idx>=1440;total+=block.size
  mr=cp.masked_case(obj,r);art=mr.artificial_mask.eq(0)
  assert mr.masked_data.where(art).isna().all().all()
  assert not (art&obj['data_raw'].isna()).any().any()
  pd.testing.assert_frame_equal(mr.masked_data.where(~art),obj['data'].where(~art))
 report={'version':'clean-20260911','train_sites':len(train),'test_sites':len(test),'site_ids_disjoint':True,'fixed_scalers_recomputed_from_training_prefixes':True,'mask_jobs_checked':len(manifest),'hidden_truth_occurrences':total,'all_hidden_truth_genuinely_observed':True,'all_masks_hide_exactly_requested_values':True,'mask_manifest_sha256':hashlib.sha256((cp.OUT/'mask_manifest.csv').read_bytes()).hexdigest(),'limitations':['Retrospective post-gap and contemporaneous training-greenhouse observations are permitted inputs.','Evaluation greenhouses were previously used for exploratory model/context/figure selection; this is not a newly untouched external test set.','Pretraining corpus membership and physical near-duplicate greenhouse identity are not established by these checks.']}
 (cp.OUT/'protocol_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()

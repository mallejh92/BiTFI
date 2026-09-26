"""Describe actual reference availability under each masked target input."""
import json,time
import numpy as np,pandas as pd
import clean_protocol as cp

def main():
 bank=cp.CleanNeighborBank();manifest=pd.read_csv(cp.OUT/'mask_manifest.csv');sites={n:cp.site(n) for n in manifest.greenhouse.unique()};records=[];t=time.time()
 for i,r in enumerate(manifest.itertuples()):
  o=sites[r.greenhouse];mr=cp.masked_case(o,r)
  for v in r.masked_vars.split(','):
   y=mr.masked_data[v].to_numpy(np.float32);obs=mr.effective_mask[v].eq(1).to_numpy();_,names=bank.select(v,o['data'].index,y,obs,k=3)
   records.append(dict(case_id=int(r.case_id),greenhouse=r.greenhouse,variable=v,n_references=len(names),references=','.join(names)))
  if i%400==0:print(i,'/',len(manifest),round(time.time()-t),'s',flush=True)
 out=cp.OUT/'quality';pd.DataFrame(records).to_csv(out/'reference_availability.csv',index=False);(out/'reference_complete.json').write_text(json.dumps(dict(cases=len(manifest),variable_cases=len(records),seconds=time.time()-t),indent=2))
if __name__=='__main__':main()

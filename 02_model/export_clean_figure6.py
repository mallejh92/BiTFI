"""Same fixed Figure 6 illustration, reevaluated with the clean protocol."""
import argparse,json
import numpy as np,pandas as pd,torch
from types import SimpleNamespace
import clean_protocol as cp
from run_clean_evaluation import build,infer
p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
s=json.loads((cp.ROOT/'03_result/figure6_common_window/selection.json').read_text());obj=cp.site(s['greenhouse']);df=obj['data'];gs=df.index.get_loc(pd.Timestamp(s['start']));ge=df.index.get_loc(pd.Timestamp(s['end']));assert obj['data_raw'][cp.COLS].iloc[gs:ge+1].notna().all().all()
m=build(a.model);rows=[]
if 'TimesFM3' in a.model:torch.set_float32_matmul_precision('highest')
for sc,sets in [('A',[[c] for c in cp.COLS]),('B',[['Tin','RH','CO2']]),('C',[cp.COLS])]:
 for vs in sets:
  row=SimpleNamespace(masked_vars=','.join(vs),start_idx=gs,end_idx=ge,gap_length_h=72,repeat=s['repeat'],scenario=sc)
  with torch.inference_mode():mr,pred=infer(m,a.model,obj,row)
  lo=max(0,gs-72);hi=min(len(df),ge+25)
  for c in (vs if sc=='A' else cp.COLS):
   y=obj['scaler'][c].inverse_transform(pred[c].iloc[lo:hi].values.reshape(-1,1)).ravel()
   rows.append(pd.DataFrame(dict(model=a.model,scenario=sc,variable=c,datetime=df.index[lo:hi],hours=np.arange(lo,hi)-gs,truth=obj['data_raw'][c].iloc[lo:hi].values,prediction=y,artificial=mr.artificial_mask[c].eq(0).iloc[lo:hi].values)))
folder=cp.OUT/'figure6_common_window';folder.mkdir(exist_ok=True);name='BiTFI' if a.model=='BiTFI-TimesFM3' else a.model;pd.concat(rows).to_csv(folder/f'{name}.csv',index=False)
(folder/'selection.json').write_text(json.dumps({**s,'clean_protocol':'clean-20260911','selection_unchanged':True,'note':'Originally selected on pre-audit results. Not reselected under corrected evaluation.'},indent=2))
print('Exported',a.model,flush=True)

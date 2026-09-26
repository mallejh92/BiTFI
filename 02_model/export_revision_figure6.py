"""Export the representative SAITS configuration for the fixed Figure 6 interval."""
from types import SimpleNamespace
import json
import numpy as np,pandas as pd,torch
import clean_protocol as cp
from revision_saits_spatial import ReferenceFeatures,network,OUT
from models.imputation_models import extract_window

def main():
 torch.set_num_threads(4);torch.manual_seed(42);torch.set_float32_matmul_precision('high')
 name='SAITS-spatial';model=network();model.load_state_dict(torch.load(OUT/name/'best.pt',weights_only=True)['state_dict']);model.eval();bank=ReferenceFeatures(False)
 selection=json.loads((cp.ROOT/'03_result/figure6_common_window/selection.json').read_text());o=cp.site(selection['greenhouse']);df=o['data'];gs=df.index.get_loc(pd.Timestamp(selection['start']));ge=df.index.get_loc(pd.Timestamp(selection['end']));rows=[]
 for sc,sets in [('A',[[v] for v in cp.COLS]),('B',[['Tin','RH','CO2']]),('C',[cp.COLS])]:
  for vs in sets:
   r=SimpleNamespace(masked_vars=','.join(vs),start_idx=gs,end_idx=ge,gap_length_h=72,repeat=selection['repeat'],scenario=sc)
   mr=cp.masked_case(o,r);masked=mr.masked_data[cp.COLS];eff=mr.effective_mask[cp.COLS];art=mr.artificial_mask[cp.COLS].eq(0);x,m,_,lo,hi=extract_window(masked,eff,art);full=masked.to_numpy(np.float32);observed=eff.ne(0).to_numpy()&np.isfinite(full)
   xx,mm,_=bank.augment(selection['greenhouse'],masked.index,full,observed,lo,hi,x,m)
   with torch.inference_mode():p=model.impute({'X':torch.from_numpy(xx[None]).cuda(),'missing_mask':torch.from_numpy(mm[None].astype(np.float32)).cuda()})[0][0,:,:5].cpu().numpy()
   pred=masked.copy();block=pred.iloc[lo:hi].to_numpy(copy=True);target=art.iloc[lo:hi].to_numpy();block[target]=p[:hi-lo][target];pred.iloc[lo:hi]=block
   lo=max(0,gs-72);hi=min(len(df),ge+25)
   for v in (vs if sc=='A' else cp.COLS):
    y=o['scaler'][v].inverse_transform(pred[v].iloc[lo:hi].to_numpy().reshape(-1,1)).ravel()
    rows.append(pd.DataFrame(dict(model=name,scenario=sc,variable=v,datetime=df.index[lo:hi],hours=np.arange(lo,hi)-gs,truth=o['data_raw'][v].iloc[lo:hi].to_numpy(),prediction=y,artificial=art[v].iloc[lo:hi].to_numpy())))
 folder=cp.ROOT/'03_result/reevaluation_context_20260925/figure6_common_window';pd.concat(rows).to_csv(folder/(name+'.csv'),index=False);print('Exported fixed-interval SAITS (local + cross)')
if __name__=='__main__':main()

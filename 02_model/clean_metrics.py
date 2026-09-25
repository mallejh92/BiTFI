"""Same physical metrics as _eval_one, computed only on the masked interval."""
import numpy as np
from evaluate import _inverse_transform,_normalized_mae
from run_comparison import make_group_labels,_r
from clean_protocol import COLS

def score_fast(name,model,obj,row,mr,pred):
 gs,ge=int(row.start_idx),int(row.end_idx);sl=slice(gs,ge+1);truth=_inverse_transform(mr.ground_truth.iloc[sl][COLS],obj['scaler']);estimate=_inverse_transform(pred.iloc[sl][COLS],obj['scaler']);labels=make_group_labels(truth.index)
 if '_metric_ranges' not in obj:obj['_metric_ranges']={c:float(obj['data_raw'][c].max()-obj['data_raw'][c].min()) for c in COLS}
 base=dict(greenhouse=obj['name'],model=name,scenario=row.scenario,masked_vars=row.masked_vars,gap_length_h=int(row.gap_length_h),repeat=int(row.repeat),context_len=getattr(model,'window',getattr(model,'context_len',1440)))
 groups=[('all','all',np.ones(len(truth),bool))]
 for kind,values in labels.items():
  arr=values.astype(str).to_numpy()
  for value in values.dropna().astype(str).unique():groups.append((kind,value,arr==value))
 rows=[]
 for kind,value,mask in groups:
  for c in row.masked_vars.split(','):
   y=truth[c].to_numpy(np.float64);p=estimate[c].to_numpy(np.float64);keep=mask&np.isfinite(y)&np.isfinite(p)
   if keep.sum()<2:continue
   err=y[keep]-p[keep];mae=float(np.mean(np.abs(err)));mse=float(np.mean(err**2))
   rows.append({**base,'group_type':kind,'group_value':value,'variable':c,'MSE':_r(mse),'MAE':_r(mae),'NMAE':_r(_normalized_mae(mae,obj['_metric_ranges'][c],y[keep])),'n_eval':int(keep.sum())})
 return rows

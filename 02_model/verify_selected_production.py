"""Diagnose selected-depth production precision without changing stored predictions."""
import copy,json,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd,torch
import clean_protocol as cp
from run_clean_evaluation import build,infer_bitfi_batch,infer

RTOL=1e-5
ATOL=1e-6


def difference(a,b,rtol=RTOL,atol=ATOL):
 a=np.asarray(a);b=np.asarray(b);both=np.isfinite(a)&np.isfinite(b)
 finite_pattern=bool(np.array_equal(np.isfinite(a),np.isfinite(b)))
 delta=np.abs(a[both]-b[both])
 threshold=atol+rtol*np.abs(b[both])
 return dict(max_absolute=float(delta.max(initial=0)),mean_absolute=float(delta.mean()) if len(delta) else 0.,
             nonzero=int(np.count_nonzero(delta)),above_original_tolerance=int(np.count_nonzero(delta>threshold)),
             finite_pattern_equal=finite_pattern,bitwise_equal=bool(np.array_equal(a,b,equal_nan=True)),
             within_original_tolerance=bool(finite_pattern and np.all(delta<=threshold)),rtol=rtol,atol=atol)


def masked_difference(a,b,row,obj):
 indices=[cp.COLS.index(v) for v in row.masked_vars.split(',')]
 x=np.asarray(a)[row.start_idx:row.end_idx+1][:,indices]
 y=np.asarray(b)[row.start_idx:row.end_idx+1][:,indices]
 result=difference(x,y)
 result['physical_max_by_variable']={v:float(np.max(np.abs(x[:,i]-y[:,i]))*obj['scaler'][v].data_range_[0]) for i,v in enumerate(row.masked_vars.split(','))}
 return result


def main():
 torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.set_float32_matmul_precision('highest')
 out=cp.OUT/'verification_20260929/production';out.mkdir(parents=True,exist_ok=True)
 report_path=out/'verification.json'
 if report_path.exists():raise RuntimeError('Verification exists; preserve it before another diagnostic run')
 manifest=pd.read_csv(cp.OUT/'mask_manifest.csv')
 chosen=manifest.groupby(['scenario','gap_length_h'],sort=False).head(2)
 sites={n:cp.site(n) for n in chosen.greenhouse.unique()};rows=list(chosen.itertuples())
 before=torch.get_float32_matmul_precision();model=build('BiTFI-TimesFM3');after=torch.get_float32_matmul_precision()
 assert model.refinements==5 and model.context_len==1900
 # Construction requests high precision internally; the original production
 # entry point explicitly restored highest AFTER constructing the backbone.
 torch.set_float32_matmul_precision('highest')
 saved_path=cp.OUT/'evaluation/BiTFI-TimesFM3/predictions.npz'
 saved_hash=hashlib.sha256(saved_path.read_bytes()).hexdigest()
 saved=np.load(saved_path);old={int(k):v for k,v in zip(saved['case_id'],saved['prediction'])}
 report=dict(selected_refinements=model.refinements,context=model.context_len,cases=0,
   case_selection='First two existing masks in every scenario-duration cell; no prediction-based selection',
   device=torch.cuda.get_device_name(0),torch=str(torch.__version__),
   precision_before_build=before,precision_after_build=after,precision_used='highest',
   original_tolerance=dict(rtol=RTOL,atol=ATOL),checks=[],serial_checks=[],original_batch_checks=[],
   high_precision_control=[],stored_prediction_sha256=saved_hash,
   manifest_sha256=hashlib.sha256((cp.OUT/'mask_manifest.csv').read_bytes()).hexdigest())
 start_time=time.time();fresh_by_case={}
 def save():
  report['elapsed_s']=time.time()-start_time;report_path.write_text(json.dumps(report,indent=2))
 for start in range(0,len(rows),16):
  chunk=rows[start:start+16];examples=[(sites[r.greenhouse],r) for r in chunk]
  fresh=infer_bitfi_batch(model,examples)
  repeated=infer_bitfi_batch(model,examples)
  poison=[]
  for obj,r in examples:
   q=copy.copy(obj);q['data']=obj['data'].copy();q['data_raw']=obj['data_raw'].copy()
   for v in r.masked_vars.split(','):
    q['data'].iloc[r.start_idx:r.end_idx+1,q['data'].columns.get_loc(v)]=1e6
    q['data_raw'].iloc[r.start_idx:r.end_idx+1,q['data_raw'].columns.get_loc(v)]=1e6
   poison.append((q,r))
  altered=infer_bitfi_batch(model,poison)
  reordered=infer_bitfi_batch(model,list(reversed(examples)))[::-1]
  for (obj,r),(mr,pred),(_,repeat),(_,pp),(_,rp) in zip(examples,fresh,repeated,altered,reordered):
   fresh_by_case[int(r.case_id)]=pred
   ys=pred.iloc[r.start_idx:r.end_idx+1][cp.COLS].to_numpy(np.float32);ref=old[int(r.case_id)][:r.gap_length_h]
   cols=[cp.COLS.index(v) for v in r.masked_vars.split(',')];obs=mr.effective_mask[cp.COLS].eq(1)
   report['checks'].append(dict(case_id=int(r.case_id),scenario=r.scenario,duration=int(r.gap_length_h),old_shard=int(r.case_id%3),
    repeated_same_batch=difference(pred,repeat),hidden_target=difference(pred,pp),batch_reversal=difference(pred,rp),
    stored_masked_predictions=difference(ys[:,cols],ref[:,cols]),
    batch_reversal_masked=masked_difference(pred,rp,r,obj),
    observations=difference(pred.values[obs.values],mr.masked_data[cp.COLS].values[obs.values])))
  report['cases']=len(report['checks']);save();print('Highest precision checks',report['cases'],'/',len(rows),round(time.time()-start_time,1),'s',flush=True)
  if start==0:
   torch.set_float32_matmul_precision('high')
   high=infer_bitfi_batch(model,examples)
   high_reverse=infer_bitfi_batch(model,list(reversed(examples)))[::-1]
   torch.set_float32_matmul_precision('highest')
   for (obj,r),(_,hp),(_,hr),(_,standard) in zip(examples,high,high_reverse,fresh):
    report['high_precision_control'].append(dict(case_id=int(r.case_id),high_vs_highest=masked_difference(hp,standard,r,obj),
      high_batch_reversal=masked_difference(hp,hr,r,obj)))
   save();print('High-versus-highest control complete',round(time.time()-start_time,1),'s',flush=True)
 # Three predefined cases cover one, three and five missing target variables.
 for scenario,h in [('A',6),('B',168),('C',72)]:
  row=next(r for r in rows if r.scenario==scenario and r.gap_length_h==h);obj=sites[row.greenhouse]
  with torch.inference_mode():_,serial=infer(model,'BiTFI-TimesFM3',obj,row)
  _,singleton=infer_bitfi_batch(model,[(obj,row)])[0]
  report['serial_checks'].append(dict(case_id=int(row.case_id),scenario=scenario,duration=h,
    serial_vs_singleton_batch=difference(serial,singleton),singleton_vs_regrouped=difference(singleton,fresh_by_case[int(row.case_id)]),
    serial_vs_singleton_masked=masked_difference(serial,singleton,row,obj)))
  save();print('Serial check',scenario,h,round(time.time()-start_time,1),'s',flush=True)
 # Match one original A6000 shard batch and one original Blackwell shard batch,
 # retaining case membership/order/chunk size, to isolate regrouping from hardware.
 targets=[next(r for r in rows if r.scenario=='C' and r.case_id%3!=0),next(r for r in rows if r.scenario=='C' and r.case_id%3==0)]
 for target in targets:
  shard=int(target.case_id%3);members=list(manifest[manifest.case_id%3==shard].itertuples())
  position=next(i for i,r in enumerate(members) if r.case_id==target.case_id);lo=(position//16)*16;chunk=members[lo:lo+16]
  for r in chunk:
   if r.greenhouse not in sites:sites[r.greenhouse]=cp.site(r.greenhouse)
  outputs=infer_bitfi_batch(model,[(sites[r.greenhouse],r) for r in chunk])
  info=json.loads((cp.OUT/f'revision/refinement/shard{shard}/complete.json').read_text())
  comparisons=[]
  for r,(_,pred) in zip(chunk,outputs):
   indices=[cp.COLS.index(v) for v in r.masked_vars.split(',')]
   new=pred.iloc[r.start_idx:r.end_idx+1][cp.COLS].to_numpy(np.float32)[:,indices]
   previous=old[int(r.case_id)][:r.gap_length_h][:,indices]
   comparisons.append(dict(case_id=int(r.case_id),difference=difference(new,previous)))
  report['original_batch_checks'].append(dict(shard=shard,original_device=info['device'],batch_start=lo,cases=comparisons))
  save();print('Original batch check',shard,round(time.time()-start_time,1),'s',flush=True)
 report['original_predictions_unchanged']=hashlib.sha256(saved_path.read_bytes()).hexdigest()==saved_hash
 report['summary']={key:dict(max_absolute=max(c[key]['max_absolute'] for c in report['checks']),
    all_within_original_tolerance=all(c[key]['within_original_tolerance'] for c in report['checks']),
    all_bitwise_equal=all(c[key]['bitwise_equal'] for c in report['checks'])) for key in ['repeated_same_batch','hidden_target','batch_reversal','stored_masked_predictions','observations']}
 report['all_serial_within_original_tolerance']=all(c['serial_vs_singleton_batch']['within_original_tolerance'] for c in report['serial_checks'])
 report['observations_note']='Raw normalized inputs are stored as float64 and copied to float32 in production; observation preservation uses the unchanged original allclose tolerance.'
 report['completed']=True;save();print(json.dumps(report['summary'],indent=2),flush=True)
 if not report['original_predictions_unchanged'] or not report['summary']['hidden_target']['all_bitwise_equal'] or not report['summary']['observations']['all_within_original_tolerance']:
  raise AssertionError('Semantic integrity check failed; inspect complete diagnostic report')

if __name__=='__main__':main()

"""Refit existing AutoGluon models with recorded timestamps, on unchanged masks.

The run root must be a separate copied experiment. No baseline artifact is modified.
Optional independent-item batching is used only after all 15 scenario-duration
smoke cases agree with the serial path and pass hidden-target perturbation checks.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, platform, shutil, time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import clean_protocol as cp
from models.autogluon_model import AutoGluonImputation
from run_comparison import AG_SPECS
from run_clean_evaluation import infer, score

BASE = cp.ROOT / '03_result/reevaluation_hourly_20260927'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frame_inputs():
    frames, validation, names = [], [], []
    for r in pd.read_csv(cp.OUT / 'sites.csv').query("group=='train'").itertuples():
        o = cp.site(r.name)
        frames.append(o['data'][cp.COLS].iloc[:o['cut']].ffill().fillna(0.))
        validation.append(o['data'][cp.COLS].iloc[o['cut']:].ffill().fillna(0.))
        names.append(r.name)
    return frames, validation, names


def check_input_time_axis(model, frames, validation, names):
    metadata = []
    for kind, seq in [('train', frames), ('validation', validation)]:
        ts = model._make_global_tsdf(seq, names)
        for name, frame in zip(names, seq):
            for v in frame.columns:
                block = ts.loc[f'{name}::{v}']
                assert block.index.equals(frame.index)
                np.testing.assert_array_equal(block.target.to_numpy(), frame[v].to_numpy())
            metadata.append(dict(site=name, partition=kind, rows=len(frame),
                                 start=str(frame.index[0]), end=str(frame.index[-1])))
    for train, val in zip(frames, validation):
        assert val.index[0] - train.index[-1] == pd.Timedelta(hours=1)
    return metadata


def batch_infer(model, examples):
    outputs, jobs = [], []
    for obj, row in examples:
        mr = cp.masked_case(obj, row)
        pred = mr.masked_data[cp.COLS].copy()
        outputs.append((mr, pred))
        gs, ge = int(row.start_idx), int(row.end_idx)
        for v in row.masked_vars.split(','):
            lo = max(0, gs - model.context_len)
            context = pred[v].iloc[lo:gs].interpolate(limit_direction='both').ffill().bfill().fillna(0.).to_numpy(np.float32)
            times = pred.index[lo:gs]
            assert times[-1] + pd.Timedelta(hours=1) == pred.index[gs]
            jobs.append(dict(output=len(outputs)-1, variable=v, gs=gs, ge=ge,
                             context=context, times=times, left=ge-gs+1, chunks=[]))
    while any(j['left'] for j in jobs):
        for step in sorted({min(j['left'], model.prediction_length) for j in jobs if j['left']}):
            active = [j for j in jobs if min(j['left'], model.prediction_length) == step]
            contexts = [j['context'][-model.context_len:] for j in active]
            timestamps = [j['times'][-len(c):] for j,c in zip(active, contexts)]
            chunks = model._predict_batch(contexts, step, timestamps)
            for j, chunk in zip(active, chunks):
                j['chunks'].append(chunk)
                j['context'] = np.concatenate([j['context'], chunk])
                j['times'] = j['times'].append(pd.date_range(j['times'][-1], periods=step+1, freq=model.freq)[1:])
                j['left'] -= step
    for j in jobs:
        pred = outputs[j['output']][1]
        pred.loc[pred.index[j['gs']:j['ge']+1],j['variable']] = np.concatenate(j['chunks'])
    return outputs


def validate_paths(model, name, manifest, sites):
    sample = manifest.groupby(['scenario','gap_length_h'],sort=False).head(1)
    examples = [(sites[r.greenhouse],r) for r in sample.itertuples()]
    serial = [infer(model,name,o,r) for o,r in examples]
    batched = batch_infer(model, examples)
    checks, batch_ok = [], True
    max_diff = 0.
    for (o,r),(mr,p),(bm,bp) in zip(examples,serial,batched):
        art = mr.artificial_mask[cp.COLS].eq(0).to_numpy()
        diff = float(np.max(np.abs(p.to_numpy()[art]-bp.to_numpy()[art])))
        max_diff = max(max_diff,diff)
        ok = np.allclose(p.to_numpy()[art],bp.to_numpy()[art],rtol=1e-5,atol=1e-6)
        batch_ok = batch_ok and ok
        poisoned = copy.copy(o)
        poisoned['data'] = o['data'].mask(mr.artificial_mask.eq(0),1e6)
        poisoned['data_raw'] = o['data_raw'].mask(mr.artificial_mask.eq(0),1e6)
        _,again = infer(model,name,poisoned,r)
        np.testing.assert_allclose(p.values,again.values,rtol=1e-5,atol=1e-6,equal_nan=True)
        checks.append(dict(case_id=int(r.case_id),scenario=r.scenario,gap=int(r.gap_length_h),
                           serial_batch_agree=bool(ok),max_abs_difference=diff,poison_invariant=True))
    # Check the batch path itself rather than inferring integrity from serial tests.
    poisoned_examples=[]
    for (o,r),(mr,p) in zip(examples,serial):
        z=copy.copy(o);z['data']=o['data'].mask(mr.artificial_mask.eq(0),1e6)
        z['data_raw']=o['data_raw'].mask(mr.artificial_mask.eq(0),1e6)
        poisoned_examples.append((z,r))
    for (mr,p),(_,q) in zip(batched,batch_infer(model,poisoned_examples)):
        np.testing.assert_allclose(p.values,q.values,rtol=1e-5,atol=1e-6,equal_nan=True)
    a,b=examples[0],examples[-1]
    _,p=infer(model,name,*a);infer(model,name,*b);_,q=infer(model,name,*a)
    np.testing.assert_allclose(p.values,q.values,rtol=1e-5,atol=1e-6,equal_nan=True)
    return dict(cases=checks,serial_batch_equivalent=bool(batch_ok),max_abs_difference=max_diff,
                batch_poison_invariant=True,ABA_order_invariant=True)


def run(name, resume=False):
    assert cp.OUT.resolve()!=BASE.resolve(),'A separate run root is required'
    assert sha(cp.OUT/'mask_manifest.csv')==sha(BASE/'mask_manifest.csv')
    folder=cp.OUT/'evaluation'/name; modeldir=cp.OUT/'models'/name
    backup=cp.OUT/'before_verified'/'autogluon'/name
    if not resume:
        assert not backup.exists(),f'{backup} exists; use --resume for the same verified run'
        backup.mkdir(parents=True)
        for kind,p in [('evaluation',folder),('model',modeldir),('smoke',cp.OUT/'smoke'/name)]:
            if p.exists():shutil.move(str(p),str(backup/kind))
    folder.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(cp.OUT/'mask_manifest.csv')
    model=AutoGluonImputation(context_len=720,prediction_length=48,time_limit=600,
                             hyperparameters=copy.deepcopy(AG_SPECS[name]),name=name,save_path=modeldir)
    metadata_path=folder/'verified_training.json'
    train_start=time.time()
    if resume and metadata_path.exists():
        metadata=json.loads(metadata_path.read_text())
        assert metadata['actual_timestamps'] and metadata['training_complete']
        model=AutoGluonImputation.load(modeldir,context_len=720,prediction_length=48,
                                      time_limit=600,hyperparameters=copy.deepcopy(AG_SPECS[name]),name=name,save_path=modeldir)
    else:
        frames,validation,names=frame_inputs()
        time_checks=check_input_time_axis(model,frames,validation,names)
        model.fit_global(frames,names=names,validation_frames=validation)
        assert model.predictor is not None,'Training failed; no interpolation fallback is accepted'
        assert model.predictor.model_names(),'Training produced no fitted model; do not evaluate or report it'
        metadata=dict(model=name,actual_timestamps=True,training_complete=True,
                      training_elapsed_s=time.time()-train_start,training_budget_s=600,
                      context=720,prediction_length=48,hyperparameters=AG_SPECS[name],
                      fill='Unchanged separate-prefix/tail ffill then leading zero',known_covariates_added=False,
                      source_run=str(BASE),mask_sha256=sha(cp.OUT/'mask_manifest.csv'),
                      split_sha256=sha(cp.OUT/'split.json'),time_checks=time_checks,
                      torch_version=torch.__version__,cuda_version=torch.version.cuda,
                      float32_matmul_precision=torch.get_float32_matmul_precision(),
                      visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() and name not in ['AG-LightGBM','AG-RandomForest'] else None),
                      available_gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      computation='CPU' if name in ['AG-LightGBM','AG-RandomForest'] else 'GPU',
                      affinity=sorted(os.sched_getaffinity(0)),torch_threads=torch.get_num_threads(),
                      python=platform.python_version(),utc=datetime.now(timezone.utc).isoformat(),
                      source_sha256={f:sha(cp.ROOT/'02_model'/f) for f in ['models/autogluon_model.py','run_verified_autogluon.py']})
        metadata_path.write_text(json.dumps(metadata,indent=2))
    model.predictor.predict=partial(model.predictor.predict,use_cache=False)
    if name=='AG-RandomForest':
        for model_name in model.predictor.model_names():
            obj=model.predictor._trainer.load_model(model_name)
            if hasattr(obj,'most_recent_model'):obj=obj.most_recent_model
            obj.get_tabular_model().model.model.n_jobs=1
    sites={n:cp.site(n) for n in manifest.greenhouse.unique()}
    start=time.time()
    checks=validate_paths(model,name,manifest,sites)
    (folder/'timestamp_and_invariance_checks.json').write_text(json.dumps(checks,indent=2))
    engine='verified independent-item batch' if checks['serial_batch_equivalent'] else 'serial'
    print('VALIDATED',name,engine,checks['max_abs_difference'],flush=True)
    result=folder/'results.csv';done=set(pd.read_csv(result).case_id.unique()) if resume and result.exists() else set()
    pending=[r for r in manifest.itertuples() if r.case_id not in done]
    pred_dir=folder/'predictions';pred_dir.mkdir(exist_ok=True)
    chunk_size=32 if checks['serial_batch_equivalent'] else 1
    for st in range(0,len(pending),chunk_size):
        chunk=pending[st:st+chunk_size];examples=[(sites[r.greenhouse],r) for r in chunk]
        with torch.inference_mode():
            outputs=batch_infer(model,examples) if checks['serial_batch_equivalent'] else [infer(model,name,o,r) for o,r in examples]
        rows=[]
        for (o,r),(mr,pred) in zip(examples,outputs):
            art=mr.artificial_mask[cp.COLS].eq(0).values
            assert np.isfinite(pred.values[art]).all()
            np.testing.assert_allclose(pred.values[mr.effective_mask[cp.COLS].eq(1).values],mr.masked_data[cp.COLS].values[mr.effective_mask[cp.COLS].eq(1).values],rtol=1e-5,atol=1e-6)
            np.save(pred_dir/f'{int(r.case_id)}.npy',pred.iloc[int(r.start_idx):int(r.end_idx)+1][cp.COLS].to_numpy(np.float32))
            newrows=score(name,model,o,r,mr,pred)
            for z in newrows:z.update(case_id=int(r.case_id),protocol='verified-20260929-real-timestamps')
            rows.extend(newrows)
        pd.DataFrame(rows).to_csv(result,mode='a',header=not result.exists(),index=False)
        if st%max(chunk_size,32)==0:print(name,st+len(chunk),'/',len(pending),'elapsed',round(time.time()-start,1),flush=True)
    d=pd.read_csv(result);a=d.query("group_type=='all'")
    expected={(int(r.case_id),v) for r in manifest.itertuples() for v in r.masked_vars.split(',')}
    assert set(zip(a.case_id,a.variable))==expected and not a.duplicated(['case_id','variable']).any()
    assert len(expected)==6205 and len(manifest)==3421 and a.n_eval.sum()==345324
    s=a.assign(w=a.NMAE*a.n_eval).groupby('greenhouse')[['w','n_eval']].sum();means=s.w/s.n_eval
    summary=dict(model=name,mask_jobs=len(manifest),all_cells=len(a),rows=len(d),NMAE=float(means.mean()),
                 SD=float(means.std()),elapsed_s=time.time()-start,protocol='verified-20260929-real-timestamps',
                 inference_engine=engine,training_elapsed_s=metadata['training_elapsed_s'],
                 actual_timestamps=True,mask_sha256=sha(cp.OUT/'mask_manifest.csv'),
                 gpu=metadata['gpu'],gpu_available=bool(torch.cuda.is_available()),
                 original_NMAE=json.loads((BASE/'evaluation'/name/'complete.json').read_text())['NMAE'])
    (folder/'complete.json').write_text(json.dumps(summary,indent=2))
    print('DONE',summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--cpu-start',type=int,default=20);p.add_argument('--cpu-count',type=int,default=4);a=p.parse_args()
    os.sched_setaffinity(0,set(range(a.cpu_start,a.cpu_start+a.cpu_count)))
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.set_float32_matmul_precision('highest')
    for name in a.models.split(','):run(name,a.resume)

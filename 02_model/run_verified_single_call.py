"""Isolate the final covariate forecast's horizon, preserving Round-0 at 128 h.

Only writes verification_20260929/single_call. Canonical model/result files are
read-only. Run with .venv_tfm3 and an explicitly selected compatible GPU.
"""
from __future__ import annotations
import argparse, copy, hashlib, inspect, json, os, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
import clean_protocol as cp
from run_clean_evaluation import build, infer, score
from clean_tfm_batch_executor import infer_many

BASE_NAME = 'TimesFM3.0-COV-SPA'
CONTROL = 'TimesFM3.0-COV-SPA-single-call'
REVISION = '43046b85ec22d584a13f8098c2ed39c889e129c2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))+'\n')


def single_covariate_target(self, all_est, col_idx, gs, ge, index):
    """Same inputs as spatial.py, but request the whole merged gap once.

    Deliberately does NOT change self.horizon_len: the inherited univariate
    Round-0 stage therefore keeps its original 128-h rolling forecasts.
    """
    horizon = ge-gs+1
    start = max(0, gs-self.context_len)
    context = all_est[start:gs, col_idx].astype(np.float32)
    if len(context) == 0:
        raise ValueError('empty context')
    column = list(self._md.columns)[col_idx]
    series = self._md[column].values.astype(np.float32)
    observed = (self._mm[column].values != 0) & np.isfinite(series)
    references = self._neighbors(column, index, series, observed)
    other = [j for j in range(all_est.shape[1]) if j != col_idx]
    covariates = np.concatenate([all_est[start:gs, :][:, other].T,
                                all_est[gs:ge+1, :][:, other].T], axis=1).astype(np.float32)
    if self.use_time_covariates:
        covariates = np.concatenate([covariates, self._time_cov(index[start:ge+1])], axis=0)
    if references.shape[0] > 0:
        covariates = np.concatenate([covariates, references[:, start:ge+1]], axis=0)
        self.stats['with_spatial'] += 1
    assert covariates.shape[1] == len(context)+horizon
    out = self.tfm.predict(context=context, horizon=horizon,
                          past_future_covariates=covariates, padding_mode='edge')
    prediction = np.asarray(out.forecast, dtype=np.float32).flatten()[:horizon]
    assert len(prediction) == horizon and np.isfinite(prediction).all()
    return prediction


def as_control(model):
    result = copy.copy(model)
    # Override on the class, not with a bound instance MethodType: infer_many
    # shallow-copies model state and must bind the method to each worker copy.
    result.__class__ = type('VerifiedSingleCovariateCall', (model.__class__,),
                            {'_predict_target': single_covariate_target})
    result.name = CONTROL
    result._nb_cache = {}
    result.stats = copy.deepcopy(model.stats)
    assert result.horizon_len == model.horizon_len == 128
    assert result._round0.__func__ is model._round0.__func__
    return result


def poison(obj, row):
    changed = copy.copy(obj)
    changed['data'] = obj['data'].copy()
    changed['data_raw'] = obj['data_raw'].copy()
    for v in row.masked_vars.split(','):
        sl = slice(row.start_idx, row.end_idx+1)
        changed['data_raw'].iloc[sl, changed['data_raw'].columns.get_loc(v)] = 1e6
        changed['data'].iloc[sl, changed['data'].columns.get_loc(v)] = 1e6
    return changed


def block(prediction, row):
    return prediction.loc[prediction.index[row.start_idx:row.end_idx+1], row.masked_vars.split(',')].to_numpy()


def validate(model, rolling, manifest, sites, folder, workers):
    sample = manifest.groupby(['scenario', 'gap_length_h'], sort=False).head(1)
    serial, checks = [], []
    for row in sample.itertuples():
        obj = sites[row.greenhouse]
        with torch.inference_mode():
            mr, pred = infer(model, CONTROL, obj, row)
            _, corrupted = infer(model, CONTROL, poison(obj, row), row)
            _, primary = infer(rolling, BASE_NAME, obj, row)
        error = float(np.nanmax(np.abs(block(pred, row)-block(corrupted, row))))
        np.testing.assert_allclose(pred.to_numpy(), corrupted.to_numpy(), atol=1e-6, rtol=1e-5, equal_nan=True)
        unchanged_short = None
        if row.gap_length_h <= 128:
            unchanged_short = float(np.nanmax(np.abs(block(pred, row)-block(primary, row))))
            np.testing.assert_allclose(block(pred, row), block(primary, row), atol=1e-6, rtol=1e-5)
        serial.append((mr, pred))
        checks.append({'case_id': row.case_id, 'scenario': row.scenario, 'gap_length_h': row.gap_length_h,
                       'hidden_truth_poison_max_abs': error, 'short_gap_vs_rolling_max_abs': unchanged_short})
        print('Smoke serial/poison', row.case_id, row.scenario, row.gap_length_h, flush=True)
    examples = [(sites[r.greenhouse], r) for r in sample.itertuples()]
    batch = infer_many(model, CONTROL, examples, workers=workers)
    batch_ok = True
    for row, (_, p), (_, q), record in zip(sample.itertuples(), serial, batch, checks):
        a, b = block(p, row), block(q, row)
        delta = float(np.abs(a-b).max())
        ok = bool(np.allclose(a, b, atol=1e-6, rtol=1e-5))
        record.update(serial_batch_max_abs=delta, serial_batch_within_tolerance=ok)
        batch_ok &= ok
    # Batch hidden-truth perturbation also checks worker isolation and grouping.
    poisoned_batch = infer_many(model, CONTROL, [(poison(o, r), r) for o, r in examples], workers=workers)
    for row, (_, p), (_, q), record in zip(sample.itertuples(), batch, poisoned_batch, checks):
        a, b = block(p, row), block(q, row)
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
        record['batch_poison_max_abs'] = float(np.abs(a-b).max())
    with torch.inference_mode():
        _, first = infer(model, CONTROL, *examples[0])
        infer(model, CONTROL, *examples[-1])
        _, again = infer(model, CONTROL, *examples[0])
    np.testing.assert_allclose(first.to_numpy(), again.to_numpy(), atol=1e-6, rtol=1e-5, equal_nan=True)
    result = {'checks': checks, 'smoke_mask_cases': len(sample), 'tolerance': {'atol': 1e-6, 'rtol': 1e-5},
              'serial_batch_validated': batch_ok, 'hidden_truth_poison_invariant': True,
              'ABA_order_invariant': True, 'highest_precision_after_build': torch.get_float32_matmul_precision()}
    dump(folder/'validation.json', result)
    return batch_ok


def merged_spans(manifest, sites):
    records = []
    for row in manifest.itertuples():
        raw = sites[row.greenhouse]['data']
        for v in row.masked_vars.split(','):
            missing = raw[v].isna().to_numpy().copy()
            missing[row.start_idx:row.end_idx+1] = True
            lo, hi = row.start_idx, row.end_idx
            while lo > 0 and missing[lo-1]:
                lo -= 1
            while hi+1 < len(missing) and missing[hi+1]:
                hi += 1
            records.append({'case_id': row.case_id, 'variable': v, 'artificial_horizon': row.gap_length_h,
                            'merged_horizon': hi-lo+1, 'merged_start': lo, 'merged_end': hi,
                            'covariate_stage_skipped_no_past': lo == 0})
    return pd.DataFrame(records)


def site_scores(d):
    z = d.assign(weighted=d.NMAE*d.n_eval).groupby(['model', 'greenhouse'])[['weighted', 'n_eval']].sum()
    return (z.weighted/z.n_eval).rename('NMAE').reset_index()


def analysis(root, folder):
    frames = [pd.read_csv(folder/'results.csv')]
    for name in [BASE_NAME, 'BiTFI-TimesFM3']:
        frames.append(pd.read_csv(root/'evaluation'/name/'results.csv'))
    d = pd.concat(frames, ignore_index=True)
    d = d[d.group_type.eq('all')]
    expected = set(zip(d[d.model.eq(BASE_NAME)].case_id, d[d.model.eq(BASE_NAME)].variable))
    for _, g in d.groupby('model'):
        assert not g.duplicated(['case_id','variable']).any()
        assert set(zip(g.case_id,g.variable)) == expected
    summaries, pairs, site_frames = [], [], []
    for scope, subset in [('all', d), ('168h', d[d.gap_length_h.eq(168)]), ('up_to_72h',d[d.gap_length_h.le(72)])]:
        ss = site_scores(subset)
        site_frames.append(ss.assign(scope=scope))
        pivot = ss.pivot(index='greenhouse', columns='model', values='NMAE')
        for model in pivot:
            x = pivot[model].dropna().to_numpy()
            ix = np.random.default_rng(42).integers(len(x), size=(10000,len(x)))
            low, high = np.quantile(x[ix].mean(1), [.025,.975])
            summaries.append({'scope':scope,'model':model,'mean':x.mean(),'sd':x.std(ddof=1),
                              'ci_low':low,'ci_high':high,'n_greenhouses':len(x)})
        for target, reference in [('BiTFI-TimesFM3',CONTROL),(CONTROL,BASE_NAME),('BiTFI-TimesFM3',BASE_NAME)]:
            p = pivot[[target,reference]].dropna()
            x,y=p[target].to_numpy(),p[reference].to_numpy()
            ix=np.random.default_rng(42).integers(len(x),size=(10000,len(x)))
            low,high=np.quantile(100*(1-x[ix].mean(1)/y[ix].mean(1)),[.025,.975])
            pairs.append({'scope':scope,'model':target,'reference':reference,
                          'reduction_percent':100*(1-x.mean()/y.mean()),'ci_low':low,'ci_high':high,
                          'n_greenhouses':len(x),'greenhouses_model_lower':int((x<y).sum())})
    pd.DataFrame(summaries).to_csv(folder/'summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(folder/'paired.csv',index=False)
    pd.concat(site_frames).to_csv(folder/'site_scores.csv',index=False)
    bycase=d.pivot(index=['case_id','variable','greenhouse','gap_length_h'],columns='model',values='NMAE').reset_index()
    bycase['single_minus_rolling_NMAE']=bycase[CONTROL]-bycase[BASE_NAME]
    bycase.to_csv(folder/'case_comparison.csv',index=False)
    lines=['# Single-call covariate forecast control','','The same TimesFM3 model, weights, 1900-h context, references, masks and 128-h univariate Round-0 initialization were retained. Only the final covariate-conditioned target forecast requested the whole merged gap in one backend call. Leading gaps without prior observations retained the existing skip/fallback. Canonical results were not replaced.','',pd.DataFrame(summaries).to_string(index=False),'',pd.DataFrame(pairs).to_string(index=False),'','Intervals are paired greenhouse bootstrap sensitivities (10,000 resamples, seed 42); no new hypothesis-test family was introduced. Comparisons against stored canonical outputs can include small hardware/batching numerical differences. Up-to-72-h cases provide an unchanged-horizon numerical control.','']
    (folder/'summary.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--analysis-only',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve();cp.OUT=root
    folder=root/'verification_20260929'/'single_call';folder.mkdir(parents=True,exist_ok=True)
    if args.analysis_only:
        analysis(root,folder);return
    assert 'revision_verified_20260929' == root.name
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    os.environ.setdefault('HF_HOME',str(cp.ROOT/'03_result/model_cache/huggingface'))
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    cache=Path(os.environ['HF_HOME'])/'hub/models--google--timesfm-3.0-pytorch'
    assert (cache/'refs/main').read_text().strip()==REVISION
    manifest=pd.read_csv(root/'mask_manifest.csv')
    sites={n:cp.site(n) for n in manifest.greenhouse.unique()}
    spans=merged_spans(manifest,sites);spans.to_csv(folder/'merged_spans.csv',index=False)
    protocol={'version':'single-covariate-call-20260929-v1','started_utc':datetime.now(timezone.utc).isoformat(),
              'root':str(root),'source_primary':'TimesFM3.0-COV-SPA','control_name':CONTROL,
              'checkpoint_revision':REVISION,'context_len':1900,'round0_horizon_unchanged':128,
              'only_change':'_predict_target covariate-conditioned forecast: full merged gap in one call instead of chunks of at most128 hours',
              'unchanged':'weights, scaler, target context, local and calendar covariates, masked Round0 filling, reference selection, leading-gap fallback, masks and scoring',
              'n_mask_cases':len(manifest),'n_variable_cases':len(spans),
              'n_merged_with_natural_gap':int(spans.merged_horizon.gt(spans.artificial_horizon).sum()),
              'n_leading_covariate_skip':int(spans.covariate_stage_skipped_no_past.sum()),
              'maximum_nonleading_request':int(spans[~spans.covariate_stage_skipped_no_past].merged_horizon.max()),
              'batch_size':args.batch_size,'workers':args.workers,
              'manifest_sha256':sha(root/'mask_manifest.csv'),'script_sha256':sha(__file__),
              'checkpoint_config_sha256':sha(cache/'snapshots'/REVISION/'config.json'),
              'checkpoint_weights_sha256':sha(cache/'snapshots'/REVISION/'model.safetensors'),
              'code_sha256':{name:sha(cp.ROOT/'02_model'/name) for name in ['spatial.py','models/foundation_model.py','clean_tfm_batch_executor.py','run_clean_evaluation.py']},
              'environment':{'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,
                             'gpu':torch.cuda.get_device_name(0),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}}
    dump(folder/'protocol.json',protocol)
    rolling=build(BASE_NAME,1900)
    # Recreate this historical isolated control explicitly after the primary default changed.
    rolling.forecasting_mode = "rolling"
    torch.set_float32_matmul_precision('highest')
    assert torch.get_float32_matmul_precision()=='highest'
    model=as_control(rolling)
    start=time.time()
    batch_ok=validate(model,rolling,manifest,sites,folder,args.workers)
    protocol['batch_mode_validated']=batch_ok
    protocol['float32_matmul_precision']=torch.get_float32_matmul_precision()
    dump(folder/'protocol.json',protocol)
    if args.smoke_only:
        return
    result=folder/'results.csv';pred_dir=folder/'predictions';pred_dir.mkdir(exist_ok=True)
    done=set(pd.read_csv(result).case_id.unique()) if result.exists() else set()
    pending=[r for r in manifest.itertuples() if r.case_id not in done]
    for offset in range(0,len(pending),args.batch_size):
        chunk=pending[offset:offset+args.batch_size]
        examples=[(sites[r.greenhouse],r) for r in chunk]
        if batch_ok:
            outputs=infer_many(model,CONTROL,examples,workers=args.workers)
        else:
            with torch.inference_mode():outputs=[infer(model,CONTROL,*x) for x in examples]
        records=[]
        for row,(mr,pred) in zip(chunk,outputs):
            np.save(pred_dir/f'{row.case_id}.npy',pred.iloc[row.start_idx:row.end_idx+1][cp.COLS].to_numpy(np.float32))
            for record in score(CONTROL,model,sites[row.greenhouse],row,mr,pred):
                record.update(case_id=row.case_id,protocol='single-covariate-call-20260929-v1')
                records.append(record)
        pd.DataFrame(records).to_csv(result,mode='a',header=not result.exists(),index=False)
        print('Single covariate call',len(done)+offset+len(chunk),'/',len(manifest),'elapsed',round(time.time()-start,1),'s',flush=True)
    analysis(root,folder)
    dump(folder/'complete.json',dict(mask_cases=len(manifest),variable_cases=len(spans),elapsed_s=time.time()-start,
                                    n_new_mask_cases=len(pending),batch_mode=batch_ok,precision=torch.get_float32_matmul_precision(),
                                    completed_utc=datetime.now(timezone.utc).isoformat(),results_sha256=sha(result)))
    print('Complete',folder,flush=True)


if __name__=='__main__':
    main()

"""Audit the TimesFM2.5 complete-gap main evaluation without changing predictions."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import clean_protocol as cp


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoke():
    import torch
    from run_clean_evaluation import build, infer

    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision('high')
    model = build('TimesFM2.5')
    assert model.forecasting_mode == 'full_gap'
    manifest = pd.read_csv(cp.OUT / 'mask_manifest.csv')
    selected = manifest.groupby(['scenario', 'gap_length_h'], sort=False).head(1)
    sites = {}
    checks = []
    forecast = model.tfm.forecast
    for row in selected.itertuples():
        if row.greenhouse not in sites:
            sites[row.greenhouse] = cp.site(row.greenhouse)
        obj = sites[row.greenhouse]
        horizons = []

        def recorded(*args, **kwargs):
            horizons.append(int(kwargs['horizon']))
            return forecast(*args, **kwargs)

        model.tfm.forecast = recorded
        try:
            with torch.inference_mode():
                mr, prediction = infer(model, 'TimesFM2.5', obj, row)
        finally:
            model.tfm.forecast = forecast
        variables = row.masked_vars.split(',')
        assert horizons == [int(row.gap_length_h)] * len(variables), horizons
        contexts = [
            mr.masked_data[v].iloc[max(0, row.start_idx-model.context_len):row.start_idx]
            .interpolate(limit_direction='both').ffill().bfill().fillna(0)
            .to_numpy(np.float32)
            for v in variables
        ]
        # Supply a companion series even in the single-variable scenario.
        batch_inputs = contexts + [contexts[0][::-1].copy()]
        with torch.inference_mode():
            batched = np.asarray(forecast(horizon=int(row.gap_length_h), inputs=batch_inputs)[0])
            reversed_batch = np.asarray(forecast(horizon=int(row.gap_length_h), inputs=batch_inputs[::-1])[0])[::-1]
        expected = prediction[variables].iloc[row.start_idx:row.end_idx+1].to_numpy().T
        np.testing.assert_allclose(batched[:len(variables)], expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(batched, reversed_batch, rtol=1e-5, atol=1e-6)
        checks.append(dict(
            case_id=int(row.case_id), scenario=row.scenario, gap_h=int(row.gap_length_h),
            requested_horizons=horizons,
            serial_batch_max_abs=float(np.max(np.abs(batched[:len(variables)]-expected))),
            batch_order_max_abs=float(np.max(np.abs(batched-reversed_batch))),
        ))
    folder = cp.OUT / 'verification_fullgap_20260929' / 'TimesFM2.5'
    folder.mkdir(parents=True, exist_ok=True)
    report = dict(
        model='TimesFM2.5', context=int(model.context_len),
        forecasting_mode=model.forecasting_mode,
        compiled_max_horizon=int(model.tfm.forecast_config.max_horizon),
        torch_version=str(torch.__version__), gpu=torch.cuda.get_device_name(),
        checks=checks, single_call_per_target=True,
        serial_batch_and_order_invariance=True,
        tolerances=dict(rtol=1e-5, atol=1e-6),
    )
    (folder/'call_and_batch_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def stored():
    manifest = pd.read_csv(cp.OUT/'mask_manifest.csv')
    oldroot = cp.ROOT/'03_result/revision_verified_20260929'
    folder = cp.OUT/'evaluation/TimesFM2.5'
    assert {int(p.stem) for p in (folder/'predictions').glob('*.npy')} == set(manifest.case_id)
    rows = pd.read_csv(folder/'results.csv').query("group_type=='all'")
    keys = {(r.case_id, v) for r in manifest.itertuples() for v in r.masked_vars.split(',')}
    assert not rows.duplicated(['case_id', 'variable']).any()
    assert set(zip(rows.case_id, rows.variable)) == keys
    indexed = rows.set_index(['case_id', 'variable'])
    uni = cp.OUT/'univariate_backbone_comparison'
    meta = pd.read_csv(uni/'cases.csv')
    uni_keys = {(int(r.mask_case_id), r.variable):i for i,r in enumerate(meta.itertuples())}
    context = json.loads((uni/'protocol.json').read_text())['context_hours']
    records = pickle.loads((uni/'inputs.pkl').read_bytes())[context]
    uni_prediction = np.load(uni/f'TimesFM2.5_univariate_ctx{context}.npz')['prediction']
    sites = {}
    mae_diff = nmae_diff = uni_diff = 0.
    input_checks = 0
    old_difference_by_gap = {}
    prediction_digest = hashlib.sha256()
    independent = []
    for row in manifest.itertuples():
        if row.greenhouse not in sites:
            sites[row.greenhouse] = cp.site(row.greenhouse)
        obj = sites[row.greenhouse]
        path = folder/'predictions'/f'{row.case_id}.npy'
        prediction_digest.update(str(int(row.case_id)).encode()+b'\0'+path.read_bytes())
        prediction = np.load(path)
        old_prediction = np.load(oldroot/'evaluation/TimesFM2.5/predictions'/f'{row.case_id}.npy')
        assert prediction.shape == (row.gap_length_h, len(cp.COLS))
        for variable in row.masked_vars.split(','):
            j = cp.COLS.index(variable)
            normalized = prediction[:,j].astype(np.float64)
            old_difference_by_gap[int(row.gap_length_h)] = max(
                old_difference_by_gap.get(int(row.gap_length_h), 0.),
                float(np.max(np.abs(normalized-old_prediction[:,j].astype(np.float64)))),
            )
            scaler = obj['scaler'][variable]
            raw = obj['data_raw'][variable]
            truth = raw.iloc[row.start_idx:row.end_idx+1].to_numpy()
            physical = normalized*scaler.data_range_[0]+scaler.data_min_[0]
            assert np.isfinite(physical).all() and np.isfinite(truth).all()
            mae = float(np.abs(physical-truth).mean())
            span = float(raw.max()-raw.min())
            nmae = mae/span
            saved = indexed.loc[(row.case_id, variable)]
            quantization = float(np.abs(np.spacing(prediction[:,j])).astype(np.float64).mean())*.5*scaler.data_range_[0]
            np.testing.assert_allclose(mae, saved.MAE, rtol=0, atol=5.0001e-7+quantization+1e-10)
            np.testing.assert_allclose(nmae, saved.NMAE, rtol=0, atol=5.0001e-7+quantization/span+1e-10)
            assert saved.n_eval == len(truth)
            mae_diff = max(mae_diff, abs(mae-saved.MAE))
            nmae_diff = max(nmae_diff, abs(nmae-saved.NMAE))
            independent.append(dict(greenhouse=row.greenhouse, nmae=nmae, n_eval=len(truth)))
            i = uni_keys[(row.case_id, variable)]
            expected_context = obj['data'][variable].iloc[max(0,row.start_idx-context):row.start_idx].interpolate(limit_direction='both').ffill().bfill().fillna(0).to_numpy(np.float32)
            np.testing.assert_array_equal(expected_context, records[i][0])
            input_checks += 1
            uni_diff = max(uni_diff, float(np.max(np.abs(normalized-uni_prediction[i,:len(truth)]))))
    d = pd.DataFrame(independent)
    scores = d.assign(w=d.nmae*d.n_eval).groupby('greenhouse')[['w','n_eval']].sum()
    scores['NMAE'] = scores.w/scores.n_eval
    old = json.loads((oldroot/'evaluation/TimesFM2.5/complete.json').read_text())
    new = json.loads((folder/'complete.json').read_text())
    source_paths = [cp.OUT/'mask_manifest.csv', cp.OUT/'scalers.pkl', cp.OUT/'context_validation/selected_contexts.json', uni/'inputs.pkl', folder/'results.csv']
    report = dict(
        model='TimesFM2.5', mask_cases=len(manifest), variable_cases=len(keys),
        prediction_files=len(list((folder/'predictions').glob('*.npy'))),
        scored_hour_occurrences=int(rows.n_eval.sum()), context_h=context,
        old_rolling_NMAE=old['NMAE'], complete_gap_NMAE=new['NMAE'],
        independently_recomputed_NMAE=float(scores.NMAE.mean()), elapsed_s=new['elapsed_s'],
        max_stored_MAE_difference=mae_diff, max_stored_NMAE_difference=nmae_diff,
        s6_context_inputs_identical_cases=input_checks,
        s6_prediction_max_abs_difference=uni_diff,
        old_prediction_max_abs_difference_by_gap=old_difference_by_gap,
        s6_comparison_note='Existing supplemental forecasts use batch_size16 and return_backcast=True; main uses batch_size1 and return_backcast=False. Input vectors and complete-gap horizons are identical; small numerical differences may remain.',
        predictions_sha256=prediction_digest.hexdigest(),
        source_sha256={str(p.relative_to(cp.OUT)):digest(p) for p in source_paths},
        all_case_keys_complete=True, independent_metric_validation_passed=True,
    )
    target = cp.OUT/'verification_fullgap_20260929/TimesFM2.5'
    target.mkdir(parents=True, exist_ok=True)
    (target/'stored_prediction_validation.json').write_text(json.dumps(report, indent=2))
    scores.to_csv(target/'independent_greenhouse_scores.csv')
    comparison=[]
    for label, root in [('rolling_128',oldroot),('complete_gap',cp.OUT)]:
        d=pd.read_csv(root/'evaluation/TimesFM2.5/results.csv').query("group_type=='all'")
        for gap, q in d.groupby('gap_length_h'):
            s=q.assign(w=q.NMAE*q.n_eval).groupby('greenhouse')[['w','n_eval']].sum()
            comparison.append(dict(call_mode=label,gap_h=int(gap),NMAE=float((s.w/s.n_eval).mean())))
    pd.DataFrame(comparison).to_csv(target/'gap_summary_before_after.csv',index=False)
    print(json.dumps(report, indent=2), flush=True)


def compare_s6():
    """Replay the largest main/S6 discrepancy under both recorded configurations."""
    import torch
    import timesfm
    from run_clean_evaluation import build

    folder=cp.OUT/'univariate_backbone_comparison'
    meta=pd.read_csv(folder/'cases.csv')
    saved=np.load(folder/'TimesFM2.5_univariate_ctx1900.npz')['prediction']
    data=pickle.loads((folder/'inputs.pkl').read_bytes())[1900]
    worst=(-1., None)
    for i,r in enumerate(meta.itertuples()):
        p=np.load(cp.OUT/'evaluation/TimesFM2.5/predictions'/f'{r.mask_case_id}.npy')[:,cp.COLS.index(r.variable)]
        err=float(np.abs(p-saved[i,:len(p)]).max())
        if err>worst[0]:worst=(err,i)
    i=worst[1]
    h=int(meta.iloc[i].gap_length_h)
    ids=np.flatnonzero(meta.gap_length_h.eq(h))
    position=int(np.flatnonzero(ids==i)[0])
    batch_ids=ids[(position//8)*8:(position//8+1)*8]
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('high')
    model=build('TimesFM2.5')
    with torch.inference_mode():
        main=np.asarray(model.tfm.forecast(horizon=h,inputs=[data[i][0]])[0])[0]
    model.tfm.compile(timesfm.ForecastConfig(
        max_context=1900,max_horizon=256,per_core_batch_size=16,
        normalize_inputs=True,use_continuous_quantile_head=True,
        force_flip_invariance=True,infer_is_positive=False,
        fix_quantile_crossing=True,return_backcast=True,
    ))
    with torch.inference_mode():
        repeated=np.asarray(model.tfm.forecast(horizon=h,inputs=[data[j][0] for j in batch_ids])[0])[:,-h:]
    expected=saved[batch_ids,:h]
    row=meta.iloc[i]
    stored_main=np.load(cp.OUT/'evaluation/TimesFM2.5/predictions'/f'{int(row.mask_case_id)}.npy')[:,cp.COLS.index(row.variable)]
    np.testing.assert_array_equal(main,stored_main)
    np.testing.assert_allclose(repeated,expected,rtol=1e-5,atol=1e-6)
    report=dict(
        max_s6_vs_main_difference=worst[0],worst_case_id=int(row.mask_case_id),
        variable=row.variable,horizon=h,original_s6_batch_ids=batch_ids.tolist(),
        original_s6_config_replay_max_abs=float(np.max(np.abs(repeated-expected))),
        new_main_config_replay_bitwise_equal=True,
        note='Both prediction series reproduced using their recorded compile/batch configuration; the cross-configuration difference is numerical, not an input or horizon mismatch.',
    )
    target=cp.OUT/'verification_fullgap_20260929/TimesFM2.5/s6_batch_numerical_check.json'
    target.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['smoke','stored','s6'])
    args=parser.parse_args()
    {'smoke':smoke,'stored':stored,'s6':compare_s6}[args.action]()

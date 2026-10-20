"""M4 (revision 2): no-history fallback for past-only adapters, applied post hoc.

The univariate past-only adapters (TimesFM3.0, TimesFM2.5 and the four AutoGluon models)
built each pre-gap context with interpolate/ffill/bfill/fillna(0). A context without any
valid observation therefore reached the model as a constant zero series, i.e. the pooled
training minimum in min-max space. `run_clean_evaluation.infer` now keeps masked-input
interpolation in that case (`no_history_fill`). This script applies the same rule to the
stored outputs of a copied run root without model inference:

  audit        count variable-cases whose past-only context has no valid observation
  verify       stub-backend comparison of infer() with and without the fallback, all masks
  rescore      replace only the affected result rows / prediction slices (six configurations)
  univariate   same rule for the matched univariate backbone comparison (Chronos2, TimesFM2.5,
               TimesFM3.0 '-UNI'; same zero-filled past-only inputs, same variable-cases)
  reaggregate  rebuild analysis/ through hourly_analysis without rewriting evaluation outputs
  compare      list every analysis value that differs from the original run root
  locate       read-only lookup of the affected manuscript strings (current -> new, line numbers)

Run from the repository root with BITFI_RUN_ROOT naming the copied revision-2 root. The
original root is only read. Backups of replaced files are kept next to them (*_before_m4.*).
"""
from __future__ import annotations
import argparse, contextlib, copy, hashlib, io, json, os, pickle, shutil, time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np, pandas as pd
import clean_protocol as cp
import run_clean_evaluation as ev
from experiment_paths import selected_context

ORIGINAL = cp.ROOT / '03_result/reevaluation_fullgap_20260929'
CONFIGS = ['TimesFM3.0', 'TimesFM2.5', 'AG-LightGBM', 'AG-RandomForest', 'AG-DeepAR', 'AG-PatchTST']
UNIVARIATE = ['Chronos2', 'TimesFM2.5', 'TimesFM3.0']
FIX = cp.OUT / 'm4_fix'
RULE = ('Past-only univariate adapters: if the pre-gap context window holds no valid observation of the '
        'target, the backend is not called and the gap keeps masked-input interpolation '
        "(masked[c].interpolate(limit_direction='both').ffill().bfill(), float32); otherwise unchanged.")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')


def guard():
    assert os.environ.get('BITFI_RUN_ROOT'), 'Set BITFI_RUN_ROOT to the copied revision-2 root'
    assert cp.OUT.resolve() != ORIGINAL.resolve(), 'The original run root is read-only'
    assert (cp.OUT / 'REVISION2_README.md').exists(), 'Expected the copied revision-2 run root'
    FIX.mkdir(exist_ok=True)


def context_lengths():
    """Actual build values: TimesFM from the saved selection, AutoGluon fixed at 720 h."""
    out = {}
    for name in CONFIGS:
        ctx = 720 if name.startswith('AG-') else selected_context(name)
        stored = pd.read_csv(cp.OUT / 'evaluation' / name / 'results_before_m4.csv' if (cp.OUT / 'evaluation' / name / 'results_before_m4.csv').exists()
                             else cp.OUT / 'evaluation' / name / 'results.csv', usecols=['context_len']).context_len.unique().tolist()
        assert stored == [ctx], (name, stored, ctx)
        out[name] = ctx
    return out


def load_inputs():
    manifest = pd.read_csv(cp.OUT / 'mask_manifest.csv')
    sites = {n: cp.site(n) for n in manifest.greenhouse.unique()}
    return manifest, sites


# ── audit ──────────────────────────────────────────────────────────────────

def audit():
    guard();ctx = context_lengths();manifest, sites = load_inputs();rows = []
    lengths = sorted(set(ctx.values()) | {1900})
    for r in manifest.itertuples():
        mr = cp.masked_case(sites[r.greenhouse], r);masked = mr.masked_data[cp.COLS];gs, ge = int(r.start_idx), int(r.end_idx)
        art = mr.artificial_mask[cp.COLS].eq(0).to_numpy()
        lo_w = max(0, min((gs + ge + 1 - 512) // 2, len(masked) - 512));hi_w = min(lo_w + 512, len(masked))
        for c in r.masked_vars.split(','):
            s = masked[c];miss = s.isna().to_numpy();lo = gs
            while lo and miss[lo - 1]:lo -= 1
            rec = dict(case_id=int(r.case_id), greenhouse=r.greenhouse, scenario=r.scenario, masked_vars=r.masked_vars, variable=c,
                       gap_length_h=int(r.gap_length_h), repeat=int(r.repeat), start_idx=gs, end_idx=ge, start_time=r.start_time,
                       merged_start=lo, leading_merged_span=bool(lo == 0), observed_before_gap=int((~miss[:gs]).sum()),
                       first_observation_after_gap=int(np.flatnonzero(~miss[ge + 1:])[0] + ge + 1) if (~miss[ge + 1:]).any() else -1,
                       chronos2_window_observations=int((~miss[max(0, gs - 1900):min(len(s), ge + 1 + 1900)]).sum()),
                       window512_target_observations=int((~miss[lo_w:hi_w] & ~art[lo_w:hi_w, cp.COLS.index(c)]).sum()))
            for L in lengths:
                rec[f'valid_context_{L}h'] = int((~miss[max(0, gs - L):gs]).sum())
            for name in CONFIGS:
                fallback = ev.no_history_fill(s, gs, ge, ctx[name])
                rec[f'no_history_{name}'] = fallback is not None
                assert (fallback is not None) == (rec[f'valid_context_{ctx[name]}h'] == 0)
            rows.append(rec)
    d = pd.DataFrame(rows);flags = [f'no_history_{n}' for n in CONFIGS]
    hit = d[d[flags].any(axis=1)].copy()
    hit.to_csv(FIX / 'no_history_cases.csv', index=False)
    summary = dict(rule=RULE, context_lengths=ctx, variable_cases=len(d), mask_cases=int(d.case_id.nunique()),
                   no_history_variable_cases={n: int(d[f'no_history_{n}'].sum()) for n in CONFIGS},
                   no_history_mask_cases=sorted(hit.case_id.unique().tolist()), greenhouses=sorted(hit.greenhouse.unique().tolist()),
                   same_cases_all_configurations=bool((d[flags].nunique(axis=1) == 1).all()),
                   no_history_by_context={str(L): int((d[f'valid_context_{L}h'] == 0).sum()) for L in lengths},
                   no_observation_anywhere_before_gap=int((d.observed_before_gap == 0).sum()),
                   leading_merged_spans=int(d.leading_merged_span.sum()),
                   no_history_are_leading_merged_spans=bool(hit.leading_merged_span.all()),
                   minimum_chronos2_both_sided_window_observations=int(d.chronos2_window_observations.min()),
                   minimum_start_idx=int(d.start_idx.min()), utc=utc())
    dump(FIX / 'no_history_audit.json', summary);print(json.dumps(summary, indent=2), flush=True)
    print(hit[['case_id', 'greenhouse', 'scenario', 'variable', 'gap_length_h', 'start_time', 'valid_context_720h', 'valid_context_1900h',
               'merged_start', 'first_observation_after_gap', 'window512_target_observations']].to_string(index=False), flush=True)
    return hit


def affected_cases():
    path = FIX / 'no_history_cases.csv'
    if not path.exists():audit()
    return pd.read_csv(path)


# ── verify ─────────────────────────────────────────────────────────────────

class StubBackend:
    """Deterministic context-sensitive stand-in for a forecaster; counts backend calls."""
    def __init__(self, context_len):
        self.context_len = context_len;self.forecasting_mode = 'full_gap';self.horizon_len = 128;self.prediction_length = 48;self.calls = 0

    def _out(self, ctx, h):
        c = np.asarray(ctx, np.float64);self.calls += 1
        return (c[-1] + np.arange(1, h + 1) * (c.mean() - c[0]) / len(c) + c.std()).astype(np.float32)

    def _forecast_one(self, ctx, h):
        return self._out(ctx, h)

    def _rolling_predict(self, item, ctx, h, timestamps=None):
        assert timestamps is not None and len(timestamps) == len(ctx)
        return self._out(ctx, h)


@contextlib.contextmanager
def legacy():
    """Original behaviour: no fallback (the only change to infer() is this branch)."""
    original = ev.no_history_fill;ev.no_history_fill = lambda *args, **kwargs: None
    try:yield
    finally:ev.no_history_fill = original


def poison(obj, row):
    z = copy.copy(obj);a = cp.masked_case(obj, row).artificial_mask.eq(0)
    z['data'] = obj['data'].mask(a, 1e6);z['data_raw'] = obj['data_raw'].mask(a, 1e6);return z


def verify():
    guard();ctx = context_lengths();manifest, sites = load_inputs();hit = affected_cases()
    report = dict(rule=RULE, method='Stub backend; modified infer() versus infer() with the fallback disabled (original code path)', configurations={})
    for name in CONFIGS:
        target = {(int(r.case_id), r.variable) for r in hit[hit[f'no_history_{name}']].itertuples()}
        new, old = StubBackend(ctx[name]), StubBackend(ctx[name]);changed = set();identical = 0;start = time.time()
        for r in manifest.itertuples():
            obj = sites[r.greenhouse];gs, ge = int(r.start_idx), int(r.end_idx)
            mr, p = ev.infer(new, name, obj, r)
            with legacy():_, q = ev.infer(old, name, obj, r)
            a, b = p[cp.COLS].to_numpy(), q[cp.COLS].to_numpy()
            for c in r.masked_vars.split(','):
                j = cp.COLS.index(c)
                if (int(r.case_id), c) in target:
                    expected = mr.masked_data[c].interpolate(limit_direction='both').ffill().bfill().iloc[gs:ge + 1].to_numpy(np.float32)
                    np.testing.assert_array_equal(a[gs:ge + 1, j], expected);changed.add((int(r.case_id), c))
                    a[gs:ge + 1, j] = b[gs:ge + 1, j]  # exclude from the identity check below
                else:identical += 1
            assert np.array_equal(a, b, equal_nan=True), (name, r.case_id)
        assert changed == target and old.calls - new.calls == len(target), (name, len(changed), old.calls, new.calls)
        poison_ok = []
        for cid in sorted({c for c, _ in target}):
            r = next(manifest[manifest.case_id.eq(cid)].itertuples());obj = sites[r.greenhouse]
            _, p = ev.infer(StubBackend(ctx[name]), name, obj, r);_, q = ev.infer(StubBackend(ctx[name]), name, poison(obj, r), r)
            assert np.array_equal(p.to_numpy(), q.to_numpy(), equal_nan=True);poison_ok.append(int(cid))
        report['configurations'][name] = dict(context_len=ctx[name], mask_cases=len(manifest), variable_cases_identical=identical,
                                              variable_cases_changed=len(changed), backend_calls_original=old.calls, backend_calls_modified=new.calls,
                                              changed_cases_equal_masked_input_interpolation=True, all_other_predictions_bitwise_identical=True,
                                              hidden_truth_poison_invariant_cases=poison_ok, seconds=round(time.time() - start, 1))
        print(name, report['configurations'][name], flush=True)
    report['utc'] = utc();dump(FIX / 'verify_stub_equivalence.json', report);return report


# ── rescore ────────────────────────────────────────────────────────────────

def read_lines(path):
    data = Path(path).read_bytes();assert data.endswith(b'\n') and b'\r' not in data
    return data[:-1].split(b'\n')


def write_lines(path, lines):
    tmp = Path(path).with_name(Path(path).name + '.m4tmp');tmp.write_bytes(b'\n'.join(lines) + b'\n');os.replace(tmp, path)


def csv_lines(frame):
    text = frame.to_csv(index=False, header=False);assert text.endswith('\n')
    return [x.encode() for x in text[:-1].split('\n')]


def save_npy(path, array):
    tmp = Path(path).with_name(Path(path).name + '.m4tmp')
    with open(tmp, 'wb') as f:np.save(f, array)
    os.replace(tmp, path)


def site_mean(d):
    a = d[d.group_type == 'all'];s = a.assign(w=a.NMAE * a.n_eval).groupby('greenhouse')[['w', 'n_eval']].sum();scores = s.w / s.n_eval
    return float(scores.mean()), float(scores.std())


def rescore():
    guard();ctx = context_lengths();manifest, sites = load_inputs();hit = affected_cases();records = [];summary = {}
    for name in CONFIGS:
        folder = cp.OUT / 'evaluation' / name;res = folder / 'results.csv';bak = folder / 'results_before_m4.csv'
        pbak = folder / 'predictions_before_m4.npz';cbak = folder / 'complete_before_m4.json'
        target = {(int(r.case_id), r.variable) for r in hit[hit[f'no_history_{name}']].itertuples()};case_ids = sorted({c for c, _ in target})
        # Backups are taken once and must equal the untouched original-root files.
        if not bak.exists():shutil.copy2(res, bak)
        if not cbak.exists():shutil.copy2(folder / 'complete.json', cbak)
        assert sha(bak) == sha(ORIGINAL / 'evaluation' / name / 'results.csv') and sha(cbak) == sha(ORIGINAL / 'evaluation' / name / 'complete.json')
        if not pbak.exists():
            tmp = folder / 'predictions_before_m4.m4tmp.npz'
            np.savez_compressed(tmp, case_id=np.asarray(case_ids, np.int64), prediction=np.stack([np.load(folder / 'predictions' / f'{i}.npy') for i in case_ids]))
            os.replace(tmp, pbak)
        stored = np.load(pbak);assert stored['case_id'].tolist() == case_ids
        before = dict(zip(case_ids, stored['prediction']))
        for i in case_ids:assert np.array_equal(before[i], np.load(ORIGINAL / 'evaluation' / name / 'predictions' / f'{i}.npy'), equal_nan=True)
        lines = read_lines(bak);header = lines[0].decode().split(',');table = pd.read_csv(bak);assert len(table) == len(lines) - 1
        new_lines = list(lines);model = SimpleNamespace(context_len=ctx[name]);replaced = 0;positions = set()
        for cid in case_ids:
            r = next(manifest[manifest.case_id.eq(cid)].itertuples());obj = sites[r.greenhouse];gs, ge = int(r.start_idx), int(r.end_idx)
            mr = cp.masked_case(obj, r);masked = mr.masked_data[cp.COLS];old_block = before[cid];new_block = old_block.copy()
            for c in r.masked_vars.split(','):
                fallback = ev.no_history_fill(masked[c], gs, ge, ctx[name])
                assert (fallback is not None) == ((cid, c) in target)
                if fallback is not None:new_block[:, cp.COLS.index(c)] = fallback
            idx = np.flatnonzero(table.case_id.eq(cid).to_numpy());protocol = table.protocol.iloc[idx].unique().tolist();assert len(protocol) == 1
            formatted = {}
            for label, block in [('old', old_block), ('new', new_block)]:
                pred = masked.copy();pred.iloc[gs:ge + 1, :] = block.astype(np.float64)
                rows = ev.score(name, model, obj, r, mr, pred)
                for z in rows:z.update(case_id=int(cid), protocol=protocol[0])
                frame = pd.DataFrame(rows);assert list(frame.columns) == header;formatted[label] = (frame, csv_lines(frame))
            frame_old, old_lines = formatted['old'];frame_new, fresh = formatted['new']
            assert len(old_lines) == len(idx) == len(fresh)
            # The stored rows are reproduced byte for byte from the stored predictions.
            assert [lines[i + 1] for i in idx] == old_lines, (name, cid)
            for k, i in enumerate(idx):
                key = (cid, frame_new.variable.iloc[k])
                assert (frame_new[['group_type', 'group_value', 'variable']].iloc[k] == table[['group_type', 'group_value', 'variable']].iloc[i]).all()
                if key in target:new_lines[i + 1] = fresh[k];replaced += 1;positions.add(i + 1)
                else:assert fresh[k] == lines[i + 1]
            for c in sorted({v for k, v in target if k == cid}):
                o = frame_old[(frame_old.group_type == 'all') & (frame_old.variable == c)].iloc[0];n = frame_new[(frame_new.group_type == 'all') & (frame_new.variable == c)].iloc[0]
                fill = float(new_block[0, cp.COLS.index(c)]);raw = float(obj['scaler'][c].inverse_transform([[fill]])[0, 0])
                records.append(dict(model=name, case_id=cid, greenhouse=r.greenhouse, scenario=r.scenario, variable=c, gap_length_h=int(r.gap_length_h),
                                    context_len=ctx[name], NMAE_before=o.NMAE, NMAE_after=n.NMAE, MAE_before=o.MAE, MAE_after=n.MAE, n_eval=int(n.n_eval),
                                    fill_value_normalized=fill, fill_value_physical=raw, prediction_before_mean_normalized=float(np.mean(old_block[:, cp.COLS.index(c)]))))
            new_pred = folder / 'predictions' / f'{cid}.npy'
            if not np.array_equal(np.load(new_pred), new_block, equal_nan=True):save_npy(new_pred, new_block.astype(np.float32))
            assert np.array_equal(np.load(new_pred), new_block, equal_nan=True)
        assert replaced == sum(len(table[(table.case_id == c) & (table.variable == v)]) for c, v in target)
        if read_lines(res) != new_lines:write_lines(res, new_lines)
        # A replaced row may be byte-identical (radiation: the first later observation is 0 W m-2, the scaler minimum).
        diff = [i for i, (a, b) in enumerate(zip(read_lines(bak), read_lines(res))) if a != b];assert set(diff) <= positions and len(read_lines(res)) == len(lines)
        d = pd.read_csv(res);old = pd.read_csv(bak)
        assert len(d) == len(old) and set(zip(d.case_id, d.variable, d.group_type, d.group_value)) == set(zip(old.case_id, old.variable, old.group_type, old.group_value)) and np.isfinite(d.NMAE).all()
        nmae, sd = site_mean(d);nmae0, sd0 = site_mean(old);complete = json.loads(cbak.read_text())
        assert abs(complete['NMAE'] - nmae0) < 1e-12
        complete.update(NMAE=nmae, SD=sd, NMAE_before_m4=nmae0, SD_before_m4=sd0,
                        m4_no_history_fallback=dict(rule=RULE, variable_cases=sorted([list(map(str, k)) for k in target]), mask_cases=case_ids,
                                                    replaced_result_rows=replaced, changed_result_lines=len(diff), script='02_model/revision2_m4_fix.py rescore',
                                                    backups=['results_before_m4.csv', 'predictions_before_m4.npz', 'complete_before_m4.json'], utc=utc()))
        dump(folder / 'complete.json', complete)
        summary[name] = dict(context_len=ctx[name], variable_cases=len(target), mask_cases=case_ids, replaced_rows=replaced, changed_lines=len(diff), NMAE_before=nmae0, NMAE_after=nmae,
                             SD_before=sd0, SD_after=sd, results_sha256_before=sha(bak), results_sha256_after=sha(res))
        print(name, summary[name], flush=True)
    pd.DataFrame(records).to_csv(FIX / 'before_after_cases.csv', index=False);dump(FIX / 'rescore.json', dict(rule=RULE, configurations=summary, utc=utc()))
    return summary


# ── univariate backbone comparison (same rule, same variable-cases) ───────

def legacy_parse(value):
    """Value obtained by pandas' default (non round-trip) float parser for a written float."""
    return float(pd.read_csv(io.StringIO(f'v\n{value!r}\n')).v.iloc[0])


def univariate():
    guard();folder = cp.OUT / 'univariate_backbone_comparison';protocol = json.loads((folder / 'protocol.json').read_text());L = int(protocol['context_hours'])
    meta = pd.read_csv(folder / 'cases.csv');assert sha(folder / 'inputs.pkl') == protocol['input_sha256']
    with (folder / 'inputs.pkl').open('rb') as f:records = pickle.load(f)[L]
    manifest, sites = load_inputs();target = {};fills = {}
    for mask_id, group in meta.groupby('mask_case_id'):
        r = next(manifest[manifest.case_id.eq(mask_id)].itertuples());mr = cp.masked_case(sites[r.greenhouse], r);masked = mr.masked_data[cp.COLS]
        for m in group.itertuples():
            fallback = ev.no_history_fill(masked[m.variable], int(r.start_idx), int(r.end_idx), L)
            if fallback is not None:
                x, _, y, factor = records[m.case_id]
                assert len(x) == min(int(r.start_idx), L) and np.all(x == 0), 'Expected the zero-filled legacy context'
                target[int(m.case_id)] = (int(mask_id), m.variable);fills[int(m.case_id)] = fallback
    hit = affected_cases();assert {v for v in target.values()} == {(int(r.case_id), r.variable) for r in hit.itertuples()}
    out = [];report = {}
    for name in UNIVARIATE:
        csv = folder / f'{name}_univariate_ctx{L}.csv';npz = folder / f'{name}_univariate_ctx{L}.npz';res = folder / f'{name}_results.csv'
        backups = {p: p.with_name(p.stem + '_before_m4' + p.suffix) for p in [csv, npz, res]}
        for p, b in backups.items():
            if not b.exists():shutil.copy2(p, b)
            assert sha(b) == sha(ORIGINAL / 'univariate_backbone_comparison' / p.name)
        pred = np.load(backups[npz])['prediction'];new_pred = pred.copy();nmae = {};max_legacy_diff = 0.;changed_lines = {}
        for i, (mask_id, v) in target.items():
            x, _, y, factor = records[i];old = float(np.mean(np.abs(pred[i, :len(y)] - y)) * factor)
            new_pred[i, :len(y)] = fills[i];new = float(np.mean(np.abs(new_pred[i, :len(y)] - y)) * factor);nmae[i] = (old, new)
        for p in [csv, res]:
            lines = read_lines(backups[p]);table = pd.read_csv(backups[p], float_precision='round_trip');assert len(table) == len(lines) - 1;new_lines = list(lines)
            for i, (old, new) in nmae.items():
                # *_results.csv is hourly_auxiliary's copy of the ctx CSV after a default-parser read.
                if p == res:old, new = legacy_parse(old), legacy_parse(new)
                k = int(np.flatnonzero(table.case_id.eq(i).to_numpy())[0]);row = table.iloc[[k]].copy()
                # Exact parsing: the stored legacy NMAE is recomputed bit for bit and the row reformatted byte for byte.
                assert row.NMAE.iloc[0] == old and csv_lines(row) == [lines[k + 1]], (name, p.name, i)
                max_legacy_diff = max(max_legacy_diff, abs(row.NMAE.iloc[0] - old))
                row['NMAE'] = new;new_lines[k + 1] = csv_lines(row)[0]
            if read_lines(p) != new_lines:write_lines(p, new_lines)
            spots = {int(np.flatnonzero(table.case_id.eq(i).to_numpy())[0]) + 1 for i in nmae}
            changed = {j for j, (a, b) in enumerate(zip(read_lines(backups[p]), read_lines(p))) if a != b}
            assert changed <= spots and len(read_lines(p)) == len(lines);changed_lines[p.name] = len(changed)
        stored = np.load(npz)['prediction']
        if not np.array_equal(stored, new_pred, equal_nan=True):
            tmp = folder / f'{name}_univariate_ctx{L}.m4tmp.npz';np.savez_compressed(tmp, prediction=new_pred);os.replace(tmp, npz)
        assert np.array_equal(np.load(npz)['prediction'], new_pred, equal_nan=True)
        d0 = pd.read_csv(backups[res]);d1 = pd.read_csv(res)
        report[name + '-UNI'] = dict(NMAE_before=site_mean(d0.assign(group_type='all'))[0], NMAE_after=site_mean(d1.assign(group_type='all'))[0], replaced_rows=len(nmae),
                                     legacy_recomputation_max_abs_difference=max_legacy_diff, changed_lines=changed_lines)
        for i, (old, new) in nmae.items():
            out.append(dict(model=name + '-UNI', comparison_case_id=i, case_id=target[i][0], variable=target[i][1], NMAE_before=old, NMAE_after=new))
        print(name, report[name + '-UNI'], flush=True)
    pd.DataFrame(out).to_csv(FIX / 'univariate_before_after_cases.csv', index=False)
    note = dict(rule=RULE, scope='Extension of M4: identical zero-filled past-only inputs (hourly_auxiliary.prepare) for the matched univariate backbone comparison',
                context_hours=L, comparison_case_ids=sorted(target), models=report, backups='*_before_m4.csv / *_before_m4.npz', utc=utc())
    dump(folder / 'm4_no_history_fallback.json', note);dump(FIX / 'univariate.json', note);return report


# ── reaggregate ────────────────────────────────────────────────────────────

def reaggregate(out=None):
    guard()
    import hourly_analysis as ha
    assert ha.OUT.resolve() == cp.OUT.resolve()
    if out:
        target = Path(out).resolve();target.mkdir(parents=True, exist_ok=True);ha.AN = target
    else:
        assert ha.AN.resolve() == (cp.OUT / 'analysis').resolve()
    checks = []

    def verify_merged(name, frames, parts=None):
        # Same frame as hourly_analysis.save_merged, compared with the stored merge; nothing is written.
        folder = cp.OUT / 'evaluation' / name;d = pd.concat(frames, ignore_index=True);d['model'] = name
        d['source_protocol'] = d.get('source_protocol', d.get('protocol', 'unspecified'));d['protocol'] = cp.PROTOCOL
        same_text = d.to_csv(index=False).encode() == (folder / 'results.csv').read_bytes()
        if not same_text:pd.testing.assert_frame_equal(pd.read_csv(folder / 'results.csv'), pd.read_csv(io.StringIO(d.to_csv(index=False))))
        same_pred = None
        if parts:
            ids = np.concatenate([x['case_id'] for x in parts]);idx = ids.argsort();stored = np.load(folder / 'predictions.npz')
            same_pred = bool(np.array_equal(stored['case_id'], ids[idx]) and np.array_equal(stored['prediction'], np.concatenate([x['prediction'] for x in parts])[idx], equal_nan=True))
            assert same_pred, name
        checks.append(dict(model=name, results_byte_identical=bool(same_text), predictions_identical=same_pred))
        return d
    ha.save_merged = verify_merged
    start = time.time();full = ha.collect();ha.diagnostics(full);ha.reporting_sensitivity(full)
    info = dict(analysis_dir=str(ha.AN), merged_outputs_verified_not_rewritten=checks, seconds=round(time.time() - start, 1), utc=utc())
    if not out:dump(FIX / 'reaggregate.json', info)
    print(json.dumps(info, indent=2), flush=True);return info


# ── compare ────────────────────────────────────────────────────────────────

KEYS = ['model', 'reference', 'variable', 'scenario', 'gap_length_h', 'group_value', 'greenhouse', 'stratum', 'n_references', 'refinements',
        'case_id', 'group_type', 'mask_case_id']


def compare(new_dir=None, old_dir=None):
    new_dir = Path(new_dir) if new_dir else cp.OUT / 'analysis';old_dir = Path(old_dir) if old_dir else ORIGINAL / 'analysis'
    rows = [];files = []
    for path in sorted(new_dir.iterdir()):
        old = old_dir / path.name
        if path.is_dir():files.append(dict(file=path.name + '/', status='directory (not compared)'));continue
        if not old.exists():files.append(dict(file=path.name, status='new'));continue
        if path.read_bytes() == old.read_bytes():files.append(dict(file=path.name, status='identical'));continue
        if path.suffix == '.json':
            a, b = json.loads(old.read_text()), json.loads(path.read_text())

            def walk(x, y, key=''):
                if isinstance(x, dict) and isinstance(y, dict):
                    for k in sorted(set(x) | set(y)):walk(x.get(k), y.get(k), f'{key}.{k}' if key else k)
                elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
                    for i, (p, q) in enumerate(zip(x, y)):walk(p, q, f'{key}[{i}]')
                elif x != y:rows.append(dict(file=path.name, key=key, column='', old=x, new=y, delta=(y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else None))
            walk(a, b);files.append(dict(file=path.name, status='changed'));continue
        if path.suffix != '.csv':
            import difflib
            changed = [x for x in difflib.unified_diff(old.read_text().splitlines(), path.read_text().splitlines(), lineterm='', n=0) if x[:1] in '+-' and x[:3] not in ('+++', '---')]
            for x in changed:rows.append(dict(file=path.name, key='text', column=x[0], old=x[1:] if x[0] == '-' else None, new=x[1:] if x[0] == '+' else None, delta=None))
            files.append(dict(file=path.name, status='changed', changed_cells=len(changed)));continue
        a, b = pd.read_csv(old), pd.read_csv(path)
        keys = [k for k in KEYS if k in a.columns and k in b.columns]
        values = [c for c in b.columns if c not in keys]
        if keys and not a.duplicated(keys).any() and not b.duplicated(keys).any():
            m = a.merge(b, on=keys, how='outer', suffixes=('_old', '_new'), indicator=True)
        else:
            m = a.add_suffix('_old').join(b.add_suffix('_new'));keys = [];m['_merge'] = 'both'
        n = 0
        for c in values:
            if f'{c}_old' not in m:continue
            x, y = m[f'{c}_old'], m[f'{c}_new']
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):diff = ~((x == y) | (x.isna() & y.isna()))
            else:diff = x.astype(str) != y.astype(str)
            for i in np.flatnonzero(diff.to_numpy()):
                key = '; '.join(f'{k}={m[k].iloc[i]}' for k in keys) if keys else f'row={i}'
                o, q = x.iloc[i], y.iloc[i]
                numeric = all(pd.api.types.is_numeric_dtype(z) and not pd.api.types.is_bool_dtype(z) for z in (x, y))
                rows.append(dict(file=path.name, key=key, column=c, old=o, new=q, delta=(q - o) if numeric else None));n += 1
        files.append(dict(file=path.name, status='changed', changed_cells=n, rows_only_old=int((m['_merge'] == 'left_only').sum()), rows_only_new=int((m['_merge'] == 'right_only').sum())))
    d = pd.DataFrame(rows);f = pd.DataFrame(files)
    print(f.to_string(index=False), flush=True)
    return d, f


# ── locate (read-only manuscript lookup) ───────────────────────────────────

LABEL = {'LI': 'Linear interpolation', 'MOMENT': 'MOMENT (zero-shot)', 'MOMENT-FT': 'MOMENT (head-tuned)', 'AG-PatchTST': 'PatchTST',
         'AG-DeepAR': 'DeepAR', 'AG-RandomForest': 'Random forest', 'SeasonalNaive': 'Seasonal naive', 'Spatial-Ridge': 'Spatial ridge',
         'SAITS': 'SAITS (sensors only)', 'TimesFM2.5': 'TimesFM2.5 (univariate)', 'AG-LightGBM': 'LightGBM', 'TimesFM3.0': 'TimesFM3 (univariate)',
         'Chronos2': 'Chronos 2 (multivariate)', 'TimesFM3.0-MV': 'TimesFM3 (multivariate)', 'SAITS-matched-local': 'SAITS (local)',
         'SAITS-spatial': 'SAITS (local + cross)', 'TimesFM3.0-COV': 'TimesFM3 (local)', 'TimesFM3.0-COV-SPA': 'TimesFM3 (local + cross)',
         'BiTFI-TimesFM3-fwd': 'BiTFI (TimesFM3, forward only)', 'BiTFI-Chronos2': 'BiTFI (Chronos 2)', 'BiTFI-TimesFM3': 'BiTFI (TimesFM3)'}


def locate():
    """Old/new manuscript strings from the original and revised analysis tables, with current line numbers."""
    guard();items = []
    for tag, root in [('old', ORIGINAL), ('new', cp.OUT)]:
        A = root / 'analysis';S = root / 'verification_20260929' / ('sensitivity' if tag == 'old' else 'sensitivity_m4')
        s = pd.read_csv(A / 'all_summary.csv').set_index('model');e = pd.read_csv(A / 'equal_case_summary.csv').set_index('model')
        q = pd.read_csv(A / 'constant_exclusion_summary.csv').set_index('model');mp = pd.read_csv(A / 'main_paired.csv').set_index('reference')
        cpd = pd.read_csv(A / 'component_paired.csv').set_index(['model', 'reference']);up = pd.read_csv(A / 'univariate_paired.csv').set_index('reference')
        v = pd.read_csv(A / 'all_variables.csv').pivot(index='model', columns='variable', values='mean')
        mae = pd.read_csv(A / 'all_physical_MAE.csv').pivot(index='model', columns='variable', values='mean')
        se = pd.read_csv(A / 'all_seasons.csv').pivot(index='model', columns='group_value', values='mean')
        uni = {m: pd.read_csv(root / 'univariate_backbone_comparison' / f'{m}_results.csv') for m in UNIVARIATE}
        sens = (S / 'supplementary_sensitivity_table.tex').read_text().splitlines()
        r_all = s['mean'].rank().astype(int);r_q = q['mean'].rank().astype(int);r_e = e['mean'].rank().astype(int)
        f4 = lambda x: f'{x:.4f}';f2 = lambda x: f'{x:.2f}'
        rows = {}
        L = 'AG-LightGBM';T = 'TimesFM3.0'
        rows['TFM text: LightGBM overall'] = ('TFM.tex', f'LightGBM ranked fourth at {f4(s.loc[L, "mean"])}')
        rows['TFM tab:overall LightGBM'] = ('TFM.tex', f'LightGBM & {f4(s.loc[L, "mean"])} & {f4(s.loc[L, "sd"])} & [{f4(s.loc[L, "low"])}, {f4(s.loc[L, "high"])}]')
        rows['TFM tab:variable LightGBM MAE'] = ('TFM.tex', 'LightGBM & ' + ' & '.join(f2(mae.loc[L, c]) for c in cp.COLS) + r' \\')
        red = cpd.loc[('TimesFM3.0-COV', T)]
        rows['TFM text: local vs univariate'] = ('TFM.tex', f'reduced TimesFM3 NMAE from {f4(s.loc[T, "mean"])} to {f4(s.loc["TimesFM3.0-COV", "mean"])} ({f2(red.reduction)}\\%)')
        rows['TFM tab:ladder univariate'] = ('TFM.tex', f'TimesFM3 (univariate) & Forward & --- & {f4(s.loc[T, "mean"])} & [{f4(s.loc[T, "low"])}, {f4(s.loc[T, "high"])}]')
        rows['TFM tab:ladder contrast'] = ('TFM.tex', f'TimesFM3: local vs.\\ univariate & {f2(red.reduction)} & [{f2(red.low)}, {f2(red.high)}]')
        rows['TFM discussion: Step 1 vs univariate'] = ('TFM.tex', f'(0.0501 versus {f4(s.loc[T, "mean"])})')
        um = {m: site_mean(uni[m].assign(group_type='all'))[0] for m in UNIVARIATE}
        rows['TFM discussion: matched univariate'] = ('TFM.tex', f'Chronos~2 and TimesFM3 had NMAEs of {f4(um["Chronos2"])} and {f4(um["TimesFM3.0"])}')
        rows['TFM discussion: local covariates'] = ('TFM.tex', f'local covariates reduced NMAE by {f2(red.reduction)}\\%')
        for m in ['AG-PatchTST', 'AG-DeepAR', 'AG-RandomForest', 'TimesFM2.5', 'AG-LightGBM', 'TimesFM3.0']:
            rows[f'S1 tab:all {m}'] = ('supplementary.tex', f'{LABEL[m]} & {f4(s.loc[m, "mean"])} & {f4(s.loc[m, "sd"])} & [{f4(s.loc[m, "low"])}, {f4(s.loc[m, "high"])}]')
        x = mp.loc[L];rows['S2 tab:paired LightGBM'] = ('supplementary.tex', f'LightGBM & {f2(x.reduction)} & [{f2(x.low)}, {f2(x.high)}] & {f4(x.holm_p)}')
        rows['S3 tab:seasons LightGBM'] = ('supplementary.tex', 'LightGBM & ' + ' & '.join(f4(se.loc[L, k]) for k in ['spring', 'summer', 'fall', 'winter']))
        for m in ['AG-PatchTST', 'AG-DeepAR', 'AG-RandomForest', 'SeasonalNaive', 'Spatial-Ridge', 'TimesFM2.5', 'AG-LightGBM', 'TimesFM3.0']:
            rows[f'S6 tab:quality {m}'] = ('supplementary.tex', f'{LABEL[m]} & {f4(s.loc[m, "mean"])} & {f4(q.loc[m, "mean"])} & {r_all[m]} & {r_q[m]}')
        rows['S8 tab:variables LightGBM'] = ('supplementary.tex', 'LightGBM & ' + ' & '.join(f4(v.loc[L, c]) for c in cp.COLS))
        rows['S12 tab:contrasts local vs univariate'] = ('supplementary.tex', f'TimesFM3: local vs. univariate & 10 & {f2(red.reduction)} & [{f2(red.low)}, {f2(red.high)}] & {f4(red.p)} & {f4(red.holm_p)}')
        for ref, lab in [('Chronos2-UNI', 'Chronos 2'), ('TimesFM2.5-UNI', 'TimesFM2.5')]:
            x = up.loc[ref];rows[f'S12 tab:contrasts univariate vs {lab}'] = ('supplementary.tex', f'TimesFM3 vs. {lab} & 10 & {f2(x.reduction)} & [{f2(x.low)}, {f2(x.high)}] & {f4(x.p)} & {f4(x.holm_p)}')
        for m in ['AG-PatchTST', 'Spatial-Ridge', 'SeasonalNaive', 'AG-DeepAR', 'AG-RandomForest', 'TimesFM2.5', 'AG-LightGBM', 'TimesFM3.0']:
            rows[f'S13 tab:equalcase {m}'] = ('supplementary.tex', f'{LABEL[m]} & {f4(s.loc[m, "mean"])} & {f4(e.loc[m, "mean"])} & [{f4(e.loc[m, "low"])}, {f4(e.loc[m, "high"])}] & {r_all[m]} & {r_e[m]}')
        rows['S14 tab:verified_sensitivity LightGBM'] = ('supplementary.tex', next(x for x in sens if x.startswith('LightGBM &')).split(r' \\')[0])
        items.append((tag, rows))
    old, new = items[0][1], items[1][1]
    texts = {f: (cp.ROOT / '05_thesis' / f).read_text().splitlines() for f in ['TFM.tex', 'supplementary.tex']}
    out = []
    for key in old:
        f, o = old[key];n = new[key][1]
        lines = [i + 1 for i, line in enumerate(texts[f]) if o in line]
        out.append(dict(item=key, file=f, lines=';'.join(map(str, lines)) or 'NOT FOUND', current=o, new=n, changed=o != n))
    d = pd.DataFrame(out)
    snapshot = {f: sha(cp.ROOT / '05_thesis' / f) for f in texts}
    d.to_csv(FIX / 'manuscript_locations.csv', index=False);dump(FIX / 'manuscript_snapshot.json', dict(sha256=snapshot, utc=utc()))
    pd.set_option('display.width', 250);pd.set_option('display.max_colwidth', 120)
    print(d.to_string(index=False), flush=True);return d


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['audit', 'verify', 'rescore', 'univariate', 'reaggregate', 'compare', 'locate'])
    p.add_argument('--out', help='reaggregate: write the tables to this directory instead of analysis/ (dry run)')
    p.add_argument('--new', help='compare: analysis directory to compare (default: run-root analysis/)')
    p.add_argument('--old', help='compare: reference analysis directory (default: original root)')
    p.add_argument('--save', help='compare: CSV path for the changed-value list')
    a = p.parse_args()
    if a.action == 'audit':audit()
    elif a.action == 'verify':verify()
    elif a.action == 'rescore':rescore()
    elif a.action == 'univariate':univariate()
    elif a.action == 'reaggregate':reaggregate(a.out)
    elif a.action == 'locate':locate()
    else:
        d, f = compare(a.new, a.old)
        if a.save:d.to_csv(a.save, index=False);f.to_csv(Path(a.save).with_name(Path(a.save).stem + '_files.csv'), index=False)


if __name__ == '__main__':
    main()

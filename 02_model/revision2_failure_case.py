"""Directional diagnosis of the largest BiTFI VPD error (revision 2).

Case 2727 (Test 06 = PF_0025101_01, Scenario C, 168 h from 2025-07-10 20:00) is rerun through the
revision-2 BiTFI code path (TimesFM3, 1900-h contexts, 5 synchronous passes). For every pass the
forward forecast, the time-reversed backward forecast and their fused estimate of each masked
variable are stored, so the pass and direction that move the estimate can be read directly.

Configurations (refinement covariates only; the univariate initialization is shared):
  BiTFI           local sensors + calendar + 3 references
  BiTFI-cross     calendar + 3 references (no local sensors)
  BiTFI-local     local sensors + calendar (no references)
  BiTFI-fwd       BiTFI without backward calls
  BiTFI-no<v>     BiTFI with local sensor v removed from the indoor-temperature calls only

Usage (BITFI_RUN_ROOT=<revision2 root>):
  python revision2_failure_case.py            TimesFM3 environment, CUDA_VISIBLE_DEVICES=1
      writes analysis/revision2/failure_case_2727/{components.npz,summary.csv,complete.json}
  python revision2_failure_case.py --screen   standard environment, CPU
      writes analysis/revision2/failure_case_2727/post_gap_run_screen.csv, the post hoc screen of every
      scored variable-case by the observed run of its variable immediately after the artificial gap
"""
import argparse, json, os, time

import numpy as np
import pandas as pd

import clean_protocol as cp
from revision2_bitfi_variants import MAIN, configure, register, now, sha

CASE = 2727
OUT = cp.OUT / 'analysis/revision2/failure_case_2727'
TIN = cp.COLS.index('Tin')
CONTEXT = 1900          # one-sided context of the main BiTFI configuration (Note S6)
SHORT_RUN = 48          # bitfi.MIN_SIDE_CTX, the minimum one-sided refinement context
STORED = {'BiTFI': MAIN, 'BiTFI-cross': 'BiTFI-cross', 'BiTFI-local': 'BiTFI-local', 'BiTFI-fwd': 'BiTFI-TimesFM3-fwd'}


def vpd(t, rh):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3)) * (1 - rh / 100)


def stored_prediction(model, h):
    folder = cp.OUT / 'evaluation' / model
    if (folder / 'predictions.npz').exists():
        z = np.load(folder / 'predictions.npz')
        return z['prediction'][np.where(z['case_id'] == CASE)[0][0]][:h]
    return np.load(folder / 'predictions' / f'{CASE}.npy')[:h]


def drop_local(model, variable):
    """Remove one local sensor from the covariate block of the indoor-temperature calls."""
    original, j = model._cov_block, cp.COLS.index(variable)

    def cov_block(est, others, nb, index, lo, hi):
        if TIN not in others:                      # the target of this call is indoor temperature
            others = [o for o in others if o != j]
        return original(est, others, nb, index, lo, hi)
    model._cov_block = cov_block


def run_one(label, row, obj, depth):
    from bitfi_refinement_ablation import infer_stages
    variant = label if label in ('BiTFI-cross', 'BiTFI-local') else MAIN
    m, settings = configure(variant)
    if label == 'BiTFI-fwd':
        m.bidirectional = False
    if label.startswith('BiTFI-no'):
        drop_local(m, label[len('BiTFI-no'):])
    calls = []
    fuse = m._fuse

    def recorder(f, b):
        p = fuse(f, b)
        calls.append((None if f is None else np.array(f), None if b is None else np.array(b), np.array(p)))
        return p
    m._fuse = recorder
    register(m, [(obj, row)])
    gs, ge, h = int(row.start_idx), int(row.end_idx), int(row.gap_length_h)
    comp = np.full((depth + 1, 3, len(cp.COLS), h), np.nan, np.float32)   # pass, (forward, backward, fused), variable
    final = None
    for k, result, _ in infer_stages(m, [(obj, row)], max_refinements=depth):
        pred = result[0][1][cp.COLS].to_numpy(np.float32)
        for f, b, p in calls:
            hits = [ci for ci in range(len(cp.COLS)) if np.array_equal(pred[gs:ge + 1, ci], p)]
            assert len(hits) == 1, (label, k, hits)
            ci = hits[0]
            comp[k, 2, ci] = p
            if f is not None: comp[k, 0, ci] = f
            if b is not None: comp[k, 1, ci] = b
        assert np.isfinite(comp[k, 2]).all(), (label, k)
        calls.clear()
        final = pred[gs:ge + 1]
    m.bank.clear()
    return comp, final, settings


def main():
    import torch
    torch.manual_seed(42); np.random.seed(42)
    row = next(pd.read_csv(cp.OUT / 'mask_manifest.csv').query('case_id == @CASE').itertuples())
    obj = cp.site(row.greenhouse)
    depth = json.loads((cp.OUT / 'refinement_validation/selection.json').read_text())['selected_refinements']
    gs, ge, h = int(row.start_idx), int(row.end_idx), int(row.gap_length_h)
    sc = obj['scaler']
    phys = lambda a, ci: sc[cp.COLS[ci]].inverse_transform(np.asarray(a, np.float64).reshape(-1, 1)).ravel()
    truth = obj['data_raw'][cp.COLS].iloc[gs:ge + 1].to_numpy(np.float64)
    labels = ['BiTFI', 'BiTFI-cross', 'BiTFI-local', 'BiTFI-fwd'] + [f'BiTFI-no{v}' for v in cp.COLS if v != 'Tin']
    started, t0 = now(), time.time()
    components, rows, repro = {}, [], {}
    for label in labels:
        comp, final, settings = run_one(label, row, obj, depth)
        components[label] = comp
        if label in STORED:
            stored = stored_prediction(STORED[label], h)
            repro[label] = dict(stored=STORED[label], max_abs_normalized=float(np.abs(final - stored).max()),
                                within_tolerance=bool(np.allclose(final, stored, rtol=1e-5, atol=1e-5)))
        for k in range(depth + 1):
            est = {c: phys(comp[k, 2, ci], ci) for ci, c in enumerate(cp.COLS)}
            r = dict(configuration=label, refinement_pass=k)
            for ci, c in enumerate(cp.COLS):
                r[f'{c}_mean'] = est[c].mean(); r[f'{c}_MAE'] = np.nanmean(np.abs(est[c] - truth[:, ci]))
                for d, name in [(0, 'forward'), (1, 'backward')]:
                    if np.isfinite(comp[k, d, ci]).all():
                        y = phys(comp[k, d, ci], ci)
                        r[f'{c}_{name}_mean'] = y.mean(); r[f'{c}_{name}_last24_mean'] = y[-24:].mean()
            v, vt = vpd(est['Tin'], est['RH']), vpd(truth[:, TIN], truth[:, cp.COLS.index('RH')])
            r.update(VPD_mean=np.nanmean(v), VPD_MAE=np.nanmean(np.abs(v - vt)), Tin_last24_mean=est['Tin'][-24:].mean())
            rows.append(r)
        print(label, 'done', round(time.time() - t0), 's', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'components.npz', configurations=np.array(labels, dtype="U16"), variables=np.array(cp.COLS, dtype="U8"),
                        directions=np.array(["forward", "backward", "fused"], dtype="U8"),
                        time=np.array([str(t) for t in obj['data'].index[gs:ge + 1]], dtype='U19'),
                        **{k.replace('-', '_'): v for k, v in components.items()})
    pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False)
    info = dict(case_id=CASE, greenhouse=row.greenhouse, scenario=row.scenario, gap_length_h=h,
                start_time=row.start_time, end_time=row.end_time, refinements=depth, configurations=labels,
                units='components.npz in normalized units (fixed training-prefix scalers); summary.csv in physical units',
                reproduction=repro, started=started, finished=now(), elapsed_s=time.time() - t0,
                device=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                run_root=str(cp.OUT), code_sha256={f: sha(os.path.join(os.path.dirname(__file__), f))
                                                   for f in ['revision2_failure_case.py', 'bitfi.py', 'bitfi_refinement_ablation.py',
                                                             'revision2_bitfi_variants.py']})
    (OUT / 'complete.json').write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(repro, indent=1))


def screen():
    """Observed run of each scored variable after its artificial gap, with BiTFI and forward-only NMAE."""
    def scores(model):
        d = pd.read_csv(cp.OUT / 'evaluation' / model / 'results.csv')
        return d[(d.group_type == 'all') & (d.model == model)]
    m = scores(MAIN).merge(scores(STORED['BiTFI-fwd'])[['case_id', 'variable', 'NMAE']], on=['case_id', 'variable'],
                           suffixes=('', '_fwd'), validate='one_to_one')
    manifest = pd.read_csv(cp.OUT / 'mask_manifest.csv').set_index('case_id')
    sites, rows = {}, []
    for r in m.itertuples():
        mr = manifest.loc[r.case_id]
        if mr.greenhouse not in sites:
            sites[mr.greenhouse] = cp.site(mr.greenhouse)['data'][cp.COLS]
        post = np.isfinite(sites[mr.greenhouse][r.variable].to_numpy()[int(mr.end_idx) + 1:int(mr.end_idx) + 1 + CONTEXT])
        run = int(np.argmin(post)) if not post.all() else len(post)
        rows.append(dict(case_id=r.case_id, variable=r.variable, first_run=run, post_len=len(post)))
    m = m.merge(pd.DataFrame(rows), on=['case_id', 'variable'], validate='one_to_one')
    m['short'] = (m.first_run < SHORT_RUN) & (m.post_len > m.first_run)   # a natural gap follows the short run
    cols = ['case_id', 'greenhouse', 'scenario', 'gap_length_h', 'variable', 'NMAE', 'NMAE_fwd', 'first_run', 'post_len', 'short']
    OUT.mkdir(parents=True, exist_ok=True)
    m[cols].to_csv(OUT / 'post_gap_run_screen.csv', index=False)
    for label, g in [('short', m[m.short]), ('other', m[~m.short])]:
        print(label, len(g), g.case_id.nunique(), round(g.NMAE.mean(), 4), round(g.NMAE_fwd.mean(), 4))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--screen', action='store_true')
    screen() if p.parse_args().screen else main()

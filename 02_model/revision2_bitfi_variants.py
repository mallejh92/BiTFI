"""BiTFI information-source and reference-selection controls (revision 2, E2).

TimesFM3 backbone, validation-selected context (1900 h), 5 synchronous refinements,
bidirectional fusion and the 3,421 main masks. The initial bidirectional univariate
stage is unchanged; only the refinement covariate block differs:

  BiTFI-local     use_spatial=False                  local sensors + calendar
  BiTFI-cross     use_local=False                    calendar + 3 references
  BiTFI-calendar  use_local=False, use_spatial=False calendar only
  BiTFI-k1        k=1                                local + calendar + top-1 reference
  BiTFI-random3   3 random eligible references       local + calendar + 3 references
  BiTFI-exclTop5  5 most used facilities excluded    local + calendar + top-3 remaining

The main configuration (BiTFI-TimesFM3) is not rerun; --repro runs it on 32 masks
and compares with the stored predictions of the original run root (read-only).

Usage (TimesFM3 environment, BITFI_RUN_ROOT=<revision2 root>):
  --repro                               reproduction check of the main configuration
  --variant V --smoke                   hidden-truth poison, rerun and order checks
  --variant V --shard i --shards 2      full run of one shard (requires a passed smoke)
  --variant V --merge                   merge shards into evaluation/V (standard env)
"""
import argparse, copy, datetime, hashlib, json, os, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import clean_protocol as cp

OUT = cp.OUT / 'revision/bitfi_variants'
ORIGINAL = cp.ROOT / '03_result/reevaluation_fullgap_20260929'
MAIN = 'BiTFI-TimesFM3'
VARIANTS = {
    'BiTFI-local': dict(use_spatial=False),
    'BiTFI-cross': dict(use_local=False),
    'BiTFI-calendar': dict(use_local=False, use_spatial=False),
    'BiTFI-k1': dict(k=1),
    'BiTFI-random3': dict(reference_mode='random'),
    'BiTFI-exclTop5': dict(reference_mode='exclude'),
}
CODE = ['bitfi.py', 'bitfi_refinement_ablation.py', 'spatial.py', 'clean_protocol.py', 'clean_metrics.py',
        'run_clean_evaluation.py', 'revision2_reference_bank.py', 'revision2_bitfi_variants.py']
TOL = dict(rtol=1e-5, atol=1e-6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def configure(variant):
    """Main BiTFI-TimesFM3 model from run_clean_evaluation.build, then the variant settings."""
    import torch
    from experiment_paths import selected_context
    from run_clean_evaluation import build
    from revision2_reference_bank import ReferenceBankVariant
    spec = {} if variant == MAIN else VARIANTS[variant]
    m = build(MAIN, selected_context('TimesFM3.0'))
    torch.set_float32_matmul_precision('highest')
    depth = json.loads((cp.OUT / 'refinement_validation/selection.json').read_text())['selected_refinements']
    m.refinements = depth
    assert m.tfm is not None and m.context_len == 1900 and m.refinements == 5 and m.bidirectional
    assert m.use_spatial and m.use_time_covariates and m.k == 3 and m.bank is not None
    exclude = ()
    if spec.get('reference_mode') == 'exclude':
        info = json.loads((OUT / 'top5_reference_facilities.json').read_text())
        assert info['recomputed_equals_stored_audit'] and info['wrapper_topk_identical_to_NeighborBank_select']
        exclude = tuple(info['top5_facilities'])
        assert len(exclude) == 5
    m.bank = ReferenceBankVariant(m.bank, spec.get('reference_mode', 'topk'), exclude=exclude, variables=cp.COLS)
    m.use_local = spec.get('use_local', True)
    m.use_spatial = spec.get('use_spatial', True)
    m.k = spec.get('k', 3)
    settings = dict(context_len=m.context_len, refinements=m.refinements, bidirectional=m.bidirectional,
                    use_local=m.use_local, use_spatial=m.use_spatial, use_time_covariates=m.use_time_covariates,
                    k=m.k, reference_mode=m.bank.mode, excluded_references=list(exclude),
                    random_seed=('default_rng([20261019, case_id, variable index in COLS])' if m.bank.mode == 'random' else None))
    # Pass-through recorder of the refinement covariate composition (local, calendar, reference rows).
    original = m._cov_block
    m._cov_signatures = Counter()

    def cov_block(est, others, nb, index, lo, hi):
        names, pf = original(est, others, nb, index, lo, hi)
        m._cov_signatures[(sum(n.startswith('v') for n in names), sum(n.startswith('t_') for n in names),
                           sum(n.startswith('nb') for n in names))] += 1
        return names, pf
    m._cov_block = cov_block
    return m, settings


def job_info(model, mr, var):
    """Refinement job of one masked variable: span, available sides and refinement directions."""
    from bitfi import MIN_SIDE_CTX
    masked = mr.masked_data[cp.COLS]
    ci = cp.COLS.index(var)
    observed = mr.effective_mask[cp.COLS].ne(0).values[:, ci] & np.isfinite(masked.values[:, ci])
    art = mr.artificial_mask[var].eq(0).values
    spans = [(gs, ge) for gs, ge in model._find_gaps(~observed) if art[gs:ge + 1].any()]
    assert len(spans) == 1, (var, spans)
    gs, ge = spans[0]
    h = ge - gs + 1
    left = min(gs, model.context_len)
    right = min(len(observed) - ge - 1, model.context_len)
    refined = h <= model._max_h()
    dirs = ('f' if left >= MIN_SIDE_CTX else '') + ('b' if model.bidirectional and right >= MIN_SIDE_CTX else '')
    return dict(job_start=int(gs), job_end=int(ge), job_hours=int(h), left_context=int(left), right_context=int(right),
                refined=bool(refined), refinement_directions=(dirs if refined else '') or 'none')


def register(model, examples):
    model.bank.clear()
    for obj, row in examples:
        mr = cp.masked_case(obj, row)
        masked = mr.masked_data[cp.COLS]
        observed = mr.effective_mask[cp.COLS].ne(0).values & np.isfinite(masked.values)
        for v in row.masked_vars.split(','):
            model.bank.register(row.case_id, row.greenhouse, v, masked.index, observed[:, cp.COLS.index(v)],
                                scenario=row.scenario, gap_length_h=int(row.gap_length_h))


REF_FIELDS = ['mode', 'k', 'status', 'n_coverage_candidates', 'n_eligible', 'n_excluded_eligible', 'n_pool',
              'n_references', 'shortfall', 'references', 'abs_r', 'eligible_ranks', 'mean_abs_r',
              'top3_references', 'top3_mean_abs_r', 'seed']


def reference_rows(model, examples, log):
    """One row per variable-case: actually used references (empty when references are off)."""
    by = {(d['case_id'], d['variable']): d for d in log}
    assert len(by) == len(log), 'more than one selection call for a variable-case'
    rows = []
    for obj, row in examples:
        mr = cp.masked_case(obj, row)
        for v in row.masked_vars.split(','):
            job = job_info(model, mr, v)
            base = dict(case_id=int(row.case_id), greenhouse=row.greenhouse, scenario=row.scenario,
                        gap_length_h=int(row.gap_length_h), variable=v, **job)
            d = by.pop((int(row.case_id), v), None)
            if model.use_spatial and job['refined']:
                assert d is not None, (row.case_id, v)
                rows.append(base | {k: d[k] for k in REF_FIELDS})
            else:
                assert d is None
                off = dict.fromkeys(REF_FIELDS)
                off.update(mode='off' if not model.use_spatial else model.bank.mode, k=0 if not model.use_spatial else model.k,
                           status='references_disabled' if not model.use_spatial else 'not_refined',
                           n_references=0, shortfall=False, references='')
                rows.append(base | off)
    assert not by, f'unmatched selection calls: {list(by)[:3]}'
    return rows


def compare_stages(a, b, label):
    worst = 0.
    for (_, x, _), (_, y, _) in zip(a, b):
        for (_, p), (_, q) in zip(x, y):
            np.testing.assert_allclose(p, q, equal_nan=True, err_msg=label, **TOL)
            worst = max(worst, float(np.nanmax(np.abs(p.to_numpy() - q.to_numpy()))))
    return worst


def poisoned(examples):
    out = []
    for o, r in examples:
        q = copy.copy(o); q['data'] = o['data'].copy(); q['data_raw'] = o['data_raw'].copy()
        for v in r.masked_vars.split(','):
            q['data'].iloc[r.start_idx:r.end_idx + 1, q['data'].columns.get_loc(v)] = 1e6
            q['data_raw'].iloc[r.start_idx:r.end_idx + 1, q['data_raw'].columns.get_loc(v)] = 1e6
        out.append((q, r))
    return out


def run(args):
    import torch
    from run_clean_evaluation import score
    from bitfi_refinement_ablation import infer_stages
    variant = MAIN if args.repro else args.variant
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    manifest = pd.read_csv(cp.OUT / 'mask_manifest.csv')
    if args.repro:
        pick = manifest.groupby(['scenario', 'gap_length_h'], sort=False).sample(2, random_state=42)
        extra = manifest.drop(pick.index).sample(32 - len(pick), random_state=42)
        manifest = pd.concat([pick, extra]).sort_values('case_id')
        folder = OUT / 'reproduction'
    elif args.smoke:
        manifest = manifest.groupby(['scenario', 'gap_length_h'], sort=False).head(1)
        folder = OUT / variant / 'smoke'
    else:
        if variant == MAIN:
            raise SystemExit('The main configuration is not rerun; use --repro')
        smoke = json.loads((OUT / variant / 'smoke/complete.json').read_text())
        assert smoke['all_checks_passed'], 'smoke checks must pass before the full run'
        manifest = manifest[manifest.case_id % args.shards == args.shard]
        folder = OUT / variant / f'shard{args.shard}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists() and not args.smoke and not args.repro:
        print('already complete', folder, flush=True); return
    started = now(); t0 = time.time()
    m, settings = configure(variant)
    depth = m.refinements
    rows = list(manifest.itertuples()); sites = {}; records = []; refs = []; checks = []
    timings = {i: 0. for i in range(depth + 1)}
    predictions = np.full((len(rows), 168, 5), np.nan, np.float32)
    torch.cuda.reset_peak_memory_stats()
    for st in range(0, len(rows), 16):
        chunk = rows[st:st + 16]; ex = []
        for r in chunk:
            if r.greenhouse not in sites:
                sites[r.greenhouse] = cp.site(r.greenhouse)
            ex.append((sites[r.greenhouse], r))
        register(m, ex)
        stages = list(infer_stages(m, ex, max_refinements=depth))
        log = m.bank.take_log()
        for k, _, dt in stages:
            timings[k] += dt
        if args.smoke or args.repro:
            check = dict(cases=len(chunk), case_ids=[int(r.case_id) for r in chunk])
            again = list(infer_stages(m, poisoned(ex), max_refinements=depth)); log_poison = m.bank.take_log()
            check['hidden_truth_poison_max_abs'] = compare_stages(stages, again, 'hidden-truth poison')
            restored = list(infer_stages(m, ex, max_refinements=depth)); log_restored = m.bank.take_log()
            check['rerun_max_abs'] = compare_stages(stages, restored, 'rerun')
            reverse = list(infer_stages(m, ex[::-1], max_refinements=depth)); log_reverse = m.bank.take_log()
            reverse = [(k, b[::-1], dt) for k, b, dt in reverse]
            check['reversed_order_max_abs'] = compare_stages(stages, reverse, 'reversed order')
            key = lambda L: sorted((d['case_id'], d['variable'], d['references']) for d in L)
            assert key(log) == key(log_poison) == key(log_restored) == key(log_reverse), 'reference selection changed'
            for (o, r), (mr, pred) in zip(ex, stages[-1][1]):
                eff = mr.effective_mask[cp.COLS].eq(1).values; art = mr.artificial_mask[cp.COLS].eq(0).values
                np.testing.assert_allclose(pred.values[eff], mr.masked_data[cp.COLS].values[eff], **TOL)
                assert np.isfinite(pred.values[art]).all()
            # Variant structure: local rows only with use_local, four calendar rows, references only with use_spatial.
            for (nl, nt, nr), _ in m._cov_signatures.items():
                assert nl == (len(cp.COLS) - 1 if m.use_local else 0) and nt == 4 and nr <= (m.k if m.use_spatial else 0), (nl, nt, nr)
            for d in log:
                assert d['n_references'] == min(m.k, d['n_pool']), d
                assert not set(d['references'].split(',')) & set(m.bank.exclude), d
            check.update(hidden_target_invariant=True, execution_order_invariant=True, rerun_identical=True,
                         reference_selection_invariant=True, observed_inputs_preserved=True, finite_predictions=True,
                         covariate_structure_as_specified=True)
            checks.append(check)
        refs.extend(reference_rows(m, ex, log))
        for i, ((o, r), (mr, pred)) in enumerate(zip(ex, stages[-1][1])):
            art = mr.artificial_mask[cp.COLS].eq(0)
            assert np.isfinite(pred.values[art.values]).all()
            q = score(variant, m, o, r, mr, pred)
            for z in q:
                z.update(case_id=int(r.case_id), refinements=depth)
            records.extend(q)
            predictions[st + i, :r.gap_length_h] = pred.iloc[r.start_idx:r.end_idx + 1][cp.COLS].to_numpy(np.float32)
        pd.DataFrame(records).to_csv(folder / 'results.csv', index=False)
        print(variant, st + len(chunk), '/', len(rows), round(time.time() - t0), 's', flush=True)
    m.bank.clear()
    np.savez_compressed(folder / 'predictions.npz', case_id=manifest.case_id.to_numpy(), prediction=predictions)
    ref = pd.DataFrame(refs); ref.to_csv(folder / 'references.csv', index=False)
    d = pd.DataFrame(records); a = d[d.group_type == 'all']
    expected = {(int(r.case_id), v) for r in rows for v in r.masked_vars.split(',')}
    assert set(zip(a.case_id, a.variable)) == expected and not a.duplicated(['case_id', 'variable']).any()
    refined = ref[ref.status.ne('references_disabled') & ref.status.ne('not_refined')]
    reference_summary = dict(
        variable_cases=len(ref),
        n_references_distribution={str(k): int(v) for k, v in ref.n_references.value_counts().sort_index().items()},
        n_references_by_variable={v: {str(k): int(c) for k, c in g.n_references.value_counts().sort_index().items()} for v, g in ref.groupby('variable')},
        status_counts={k: int(v) for k, v in ref.status.value_counts().items()},
        shortfall_below_k=int(ref.shortfall.sum()),
        zero_reference_cases_with_references_enabled=int((refined.n_references == 0).sum()) if len(refined) else 0,
        mean_selected_abs_r=float(refined.mean_abs_r.mean()) if len(refined) else None,
        mean_main_top3_abs_r=float(refined.top3_mean_abs_r.mean()) if len(refined) else None,
        refinement_directions={k: int(v) for k, v in ref.refinement_directions.value_counts().items()},
        not_refined_too_long=int((~ref.refined).sum()))
    info = dict(variant=variant, spec=({} if variant == MAIN else VARIANTS[variant]), settings=settings,
                purpose=('reproduction of the stored main run' if args.repro else 'smoke' if args.smoke else 'full shard'),
                shard=None if (args.smoke or args.repro) else args.shard, shards=None if (args.smoke or args.repro) else args.shards,
                cases=len(rows), variable_cases=len(ref), started=started, finished=now(), elapsed_s=time.time() - t0,
                seconds_per_stage=timings, device=torch.cuda.get_device_name(0),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=str(torch.__version__),
                matmul_precision=torch.get_float32_matmul_precision(),
                peak_gpu_memory_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                peak_gpu_memory_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                covariate_signatures={f'local{a}_calendar{b}_references{c}': int(n) for (a, b, c), n in sorted(m._cov_signatures.items())},
                references=reference_summary, protocol=cp.PROTOCOL, run_root=str(cp.OUT),
                mask_manifest_sha256=sha(cp.OUT / 'mask_manifest.csv'),
                code_sha256={f: sha(Path(__file__).parent / f) for f in CODE})
    if args.smoke or args.repro:
        info.update(checks=checks, all_checks_passed=all(c['hidden_target_invariant'] for c in checks),
                    tolerance=TOL)
    else:
        smoke = json.loads((OUT / variant / 'smoke/complete.json').read_text())
        info['smoke'] = dict(path=str(OUT / variant / 'smoke/complete.json'), all_checks_passed=smoke['all_checks_passed'],
                             cases=smoke['cases'], finished=smoke['finished'],
                             max_abs=dict(poison=max(c['hidden_truth_poison_max_abs'] for c in smoke['checks']),
                                          rerun=max(c['rerun_max_abs'] for c in smoke['checks']),
                                          reversed_order=max(c['reversed_order_max_abs'] for c in smoke['checks'])))
    if args.repro:
        info['reproduction'] = reproduction(manifest, d, predictions)
        info['all_checks_passed'] = info['all_checks_passed'] and info['reproduction']['within_tolerance']
    (folder / 'complete.json').write_text(json.dumps(info, indent=2, default=str))
    print('DONE', json.dumps({k: info[k] for k in ['variant', 'purpose', 'cases', 'elapsed_s', 'device']}), flush=True)


def reproduction(manifest, d, predictions):
    """Compare the base configuration with the stored main predictions (original root, read-only)."""
    path = ORIGINAL / 'evaluation' / MAIN / 'predictions.npz'
    stored = np.load(path); stored = {int(c): p for c, p in zip(stored['case_id'], stored['prediction'])}
    res = pd.read_csv(ORIGINAL / 'evaluation' / MAIN / 'results.csv').query("group_type=='all'")
    shard_device = {}
    for i in range(3):
        dev = json.loads((ORIGINAL / f'revision/refinement/shard{i}/complete.json').read_text())['device']
        shard_device[i] = dev
    cases = []; ok = True
    for i, r in enumerate(manifest.itertuples()):
        idx = [cp.COLS.index(v) for v in r.masked_vars.split(',')]
        x = predictions[i, :r.gap_length_h][:, idx]; y = stored[int(r.case_id)][:r.gap_length_h][:, idx]
        diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
        close = bool(np.allclose(x, y, **TOL))
        ok &= close
        full = np.allclose(predictions[i, :r.gap_length_h], stored[int(r.case_id)][:r.gap_length_h], equal_nan=True, **TOL)
        cases.append(dict(case_id=int(r.case_id), greenhouse=r.greenhouse, scenario=r.scenario, gap=int(r.gap_length_h),
                          original_shard=int(r.case_id) % 3, original_device=shard_device[int(r.case_id) % 3],
                          masked_max_abs=float(diff.max()), within_tolerance=close, all_columns_within_tolerance=bool(full)))
    a = d[d.group_type == 'all'].merge(res[['case_id', 'variable', 'NMAE', 'MAE']], on=['case_id', 'variable'],
                                         suffixes=('_new', '_stored'), validate='one_to_one')
    return dict(stored_predictions=str(path), stored_predictions_sha256=sha(path), cases=len(cases),
                variable_cases=len(a), within_tolerance=bool(ok), tolerance=TOL,
                max_abs_prediction_difference=max(c['masked_max_abs'] for c in cases),
                max_abs_NMAE_difference=float((a.NMAE_new - a.NMAE_stored).abs().max()),
                max_abs_MAE_difference=float((a.MAE_new - a.MAE_stored).abs().max()),
                per_case=cases)


def merge(variant):
    """Merge both shards with hourly_analysis.save_merged (same format and checks)."""
    import fcntl
    from hourly_analysis import save_merged
    folder = OUT / variant
    with (folder / '.merge.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / 'merged.json').exists():
            print('already merged', folder, flush=True); return
        infos = [json.loads((folder / f'shard{i}/complete.json').read_text()) for i in range(2)]
        assert [x['shard'] for x in infos] == [0, 1] and all(x['shards'] == 2 for x in infos)
        frames = [pd.read_csv(folder / f'shard{i}/results.csv') for i in range(2)]
        parts = [np.load(folder / f'shard{i}/predictions.npz') for i in range(2)]
        d = save_merged(variant, frames, parts)
        ref = pd.concat([pd.read_csv(folder / f'shard{i}/references.csv') for i in range(2)], ignore_index=True)
        manifest = pd.read_csv(cp.OUT / 'mask_manifest.csv')
        expected = {(int(r.case_id), v) for r in manifest.itertuples() for v in r.masked_vars.split(',')}
        assert set(zip(ref.case_id, ref.variable)) == expected and not ref.duplicated(['case_id', 'variable']).any()
        ref.sort_values(['case_id', 'variable']).to_csv(folder / 'references.csv', index=False)
        a = d[d.group_type == 'all']; s = a.assign(w=a.NMAE * a.n_eval).groupby('greenhouse')[['w', 'n_eval']].sum()
        summary = dict(variant=variant, cases=int(a.case_id.nunique()), variable_cases=len(a), NMAE=float((s.w / s.n_eval).mean()),
                       shards=[dict(shard=x['shard'], cases=x['cases'], device=x['device'], started=x['started'],
                                    finished=x['finished'], elapsed_s=x['elapsed_s']) for x in infos],
                       code_sha256_consistent=infos[0]['code_sha256'] == infos[1]['code_sha256'],
                       results=str(cp.OUT / 'evaluation' / variant / 'results.csv'),
                       predictions=str(cp.OUT / 'evaluation' / variant / 'predictions.npz'),
                       results_sha256=sha(cp.OUT / 'evaluation' / variant / 'results.csv'),
                       predictions_sha256=sha(cp.OUT / 'evaluation' / variant / 'predictions.npz'), merged=now())
        (folder / 'merged.json').write_text(json.dumps(summary, indent=2))
        print('MERGED', json.dumps({k: summary[k] for k in ['variant', 'cases', 'variable_cases', 'NMAE']}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=list(VARIANTS))
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=2)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--repro', action='store_true')
    p.add_argument('--merge', action='store_true')
    a = p.parse_args()
    if a.merge:
        merge(a.variant)
    else:
        assert a.repro or a.variant, '--variant required'
        run(a)


if __name__ == '__main__':
    main()

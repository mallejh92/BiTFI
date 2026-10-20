"""First-pass summary of the BiTFI information-source and reference controls (revision 2, E2).

Greenhouse-level statistics with analyze_revision_experiments.summary/site_scores/compare:
hour-weighted NMAE within greenhouse, equal greenhouse weights, 10,000 greenhouse
bootstrap resamples (seed 42), paired Wilcoxon signed-rank tests over greenhouses and
Holm adjustment over the six variant comparisons as one family.

compare(s, [(variant, BiTFI-TimesFM3)]) gives reduction = 100 * (1 - variant / main);
change_pct = -reduction is the variant's NMAE change relative to BiTFI-TimesFM3
(positive = higher error than the full model). The manuscript-convention reduction of
BiTFI-TimesFM3 relative to each variant (denominator = variant) is reported alongside.
"""
import hashlib, json
import numpy as np, pandas as pd
import clean_protocol as cp
from analyze_revision_experiments import summary, site_scores, compare
from revision2_bitfi_variants import VARIANTS, MAIN, OUT as RUNS

AN = cp.OUT / 'analysis/revision2'
ORDER = [MAIN, *VARIANTS]
TOP5 = json.loads((RUNS / 'top5_reference_facilities.json').read_text())['top5_facilities']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load():
    frames = []
    for m in ORDER:
        d = pd.read_csv(cp.OUT / 'evaluation' / m / 'results.csv')
        assert set(d.model) == {m}
        frames.append(d)
    full = pd.concat(frames, ignore_index=True)
    base = full.query("group_type=='all'")
    keys = None
    for m, g in base.groupby('model'):
        k = set(zip(g.case_id, g.variable))
        assert keys is None or k == keys, m
        keys = k
        assert len(g) == 6205 and g.case_id.nunique() == 3421 and not g.duplicated(['case_id', 'variable']).any()
        assert np.isfinite(g.NMAE).all()
    return full, base


def paired(s, family):
    """Variant vs main (Holm over the family), with the change and manuscript-convention columns."""
    a = compare(s, [(v, MAIN) for v in family])
    b = compare(s, [(MAIN, v) for v in family]).set_index('reference')
    a['change_pct'] = -a.reduction; a['change_low'] = -a.high; a['change_high'] = -a.low
    a['main_reduction_vs_variant'] = a.model.map(b.reduction)
    a['main_reduction_low'] = a.model.map(b.low); a['main_reduction_high'] = a.model.map(b.high)
    p = s.pivot(index='greenhouse', columns='model', values='NMAE')
    a['variant_mean'] = a.model.map(p.mean()); a['main_mean'] = p[MAIN].mean()
    a['greenhouses_variant_worse'] = a.model.map(lambda v: int((p[v] > p[MAIN]).sum()))
    return a


def with_change(t, extra):
    keys = list(extra)
    ref = t[t.model == MAIN].set_index(keys)['mean'] if keys else t.loc[t.model == MAIN, 'mean'].iloc[0]
    if keys:
        t['main_mean'] = [ref.loc[tuple(r[k] for k in keys) if len(keys) > 1 else r[keys[0]]] for _, r in t.iterrows()]
    else:
        t['main_mean'] = ref
    t['change_vs_main_pct'] = 100 * (t['mean'] / t.main_mean - 1)
    return t


def references():
    rows = []; slots = []; cases = []
    for v in VARIANTS:
        r = pd.read_csv(RUNS / v / 'references.csv')
        r['variant'] = v; cases.append(r)
        for scope, g in [('all', r), *list(r.groupby('variable'))]:
            on = g[g.status.ne('references_disabled') & g.status.ne('not_refined')]
            ranks = on.eligible_ranks.dropna().astype(str).str.split(';').explode() if len(on) else pd.Series(dtype=str)
            ranks = pd.to_numeric(ranks[ranks != ''], errors='coerce')
            names = on.references.fillna('').str.split(',').explode() if len(on) else pd.Series(dtype=str)
            names = names[names != '']
            top3 = on.top3_references.fillna('').str.split(',') if len(on) else pd.Series(dtype=object)
            sel = on.references.fillna('').str.split(',') if len(on) else pd.Series(dtype=object)
            in_top3 = [len(set(a) & set(b) - {''}) for a, b in zip(sel, top3)]
            rows.append(dict(variant=v, variable=scope, variable_cases=len(g),
                             references_enabled=len(on),
                             **{f'n_references_{k}': int((g.n_references == k).sum()) for k in range(4)},
                             shortfall_below_k=int(g.shortfall.sum()),
                             mean_references=float(g.n_references.mean()),
                             mean_selected_abs_r=float(on.mean_abs_r.mean()) if len(on) else np.nan,
                             mean_main_top3_abs_r=float(on.top3_mean_abs_r.mean()) if len(on) else np.nan,
                             mean_abs_r_difference=float((on.mean_abs_r - on.top3_mean_abs_r).mean()) if len(on) else np.nan,
                             median_abs_r_difference=float((on.mean_abs_r - on.top3_mean_abs_r).median()) if len(on) else np.nan,
                             mean_selected_rank=float(ranks.mean()) if len(ranks) else np.nan,
                             slots=int(len(names)),
                             slots_from_main_top3=int(sum(in_top3)) if len(on) else 0,
                             slot_share_top5_facilities=float(names.isin(TOP5).mean()) if len(names) else np.nan,
                             mean_eligible=float(on.n_eligible.mean()) if len(on) else np.nan,
                             mean_pool=float(on.n_pool.mean()) if len(on) else np.nan))
        if len(names := r.references.fillna('').str.split(',').explode()):
            names = names[names != '']
            for n, c in names.value_counts().items():
                slots.append(dict(variant=v, reference=n, slots=int(c), share=float(c / len(names)), top5=n in TOP5))
    return pd.DataFrame(rows), pd.DataFrame(slots), pd.concat(cases, ignore_index=True)


def main():
    AN.mkdir(parents=True, exist_ok=True)
    full, base = load()
    s = site_scores(base)
    with_change(summary(base), []).to_csv(AN / 'bitfi_variants_summary.csv', index=False)
    with_change(summary(base, extra=['scenario']), ['scenario']).to_csv(AN / 'bitfi_variants_scenarios.csv', index=False)
    with_change(summary(base, extra=['variable']), ['variable']).to_csv(AN / 'bitfi_variants_variables.csv', index=False)
    with_change(summary(base, extra=['scenario', 'gap_length_h']), ['scenario', 'gap_length_h']).to_csv(AN / 'bitfi_variants_gaps.csv', index=False)
    with_change(summary(base, extra=['scenario', 'variable']), ['scenario', 'variable']).to_csv(AN / 'bitfi_variants_scenario_variables.csv', index=False)
    s.pivot(index='greenhouse', columns='model', values='NMAE')[ORDER].reset_index().to_csv(AN / 'bitfi_variants_sites.csv', index=False)
    paired(s, list(VARIANTS)).to_csv(AN / 'bitfi_variants_paired.csv', index=False)
    parts = []
    for sc, g in base.groupby('scenario'):
        q = paired(site_scores(g), list(VARIANTS)); q.insert(0, 'scenario', sc); parts.append(q)
    pd.concat(parts).to_csv(AN / 'bitfi_variants_paired_scenarios.csv', index=False)
    parts = []
    for var, g in base.groupby('variable'):
        q = paired(site_scores(g), list(VARIANTS)); q.insert(0, 'variable', var); parts.append(q)
    pd.concat(parts).to_csv(AN / 'bitfi_variants_paired_variables.csv', index=False)
    ref, slots, cases = references()
    ref.to_csv(AN / 'bitfi_variants_references.csv', index=False)
    slots.to_csv(AN / 'bitfi_variants_reference_slots.csv', index=False)
    # exclTop5 rows with fewer than three remaining references, against the main run on the same cells
    ex = cases[cases.variant == 'BiTFI-exclTop5'][['case_id', 'variable', 'n_references']]
    sub = base[base.model.isin([MAIN, 'BiTFI-exclTop5'])].merge(ex, on=['case_id', 'variable'], validate='many_to_one')
    sub['stratum'] = np.where(sub.n_references < 3, 'fewer_than_3_after_exclusion', 'three_after_exclusion')
    t = summary(sub, extra=['stratum'])
    t['variable_cases'] = t.apply(lambda r: int(((sub.model == r.model) & (sub.stratum == r.stratum)).sum()), axis=1)
    t.to_csv(AN / 'bitfi_variants_exclTop5_strata.csv', index=False)
    # Context: stored Step-1-only arm (bidirectional initialization, no refinement) on the same cells.
    r0 = pd.read_csv(cp.OUT / 'analysis/refinement_cases.csv').query('refinements==0 and group_type=="all"')
    assert set(zip(r0.case_id, r0.variable)) == set(zip(base.case_id, base.variable)) and set(r0.model) == {'BiTFI-refine0'}
    ctx = pd.concat([base, r0[base.columns.intersection(r0.columns)]], ignore_index=True)
    sc = site_scores(ctx)
    summary(ctx).to_csv(AN / 'bitfi_variants_with_refine0_summary.csv', index=False)
    q = compare(sc, [(v, 'BiTFI-refine0') for v in ORDER])
    q.to_csv(AN / 'bitfi_variants_vs_refine0_paired.csv', index=False)
    timing = []
    for v in VARIANTS:
        for i in range(2):
            c = json.loads((RUNS / v / f'shard{i}/complete.json').read_text())
            timing.append(dict(variant=v, shard=i, cases=c['cases'], device=c['device'], gpu=c['cuda_visible_devices'],
                               started=c['started'], finished=c['finished'], elapsed_s=c['elapsed_s'],
                               backbone_and_refinement_s=sum(c['seconds_per_stage'].values()),
                               peak_gpu_memory_reserved_gb=c['peak_gpu_memory_reserved_gb']))
    pd.DataFrame(timing).to_csv(AN / 'bitfi_variants_timing.csv', index=False)
    sources = {m: dict(results=str(cp.OUT / 'evaluation' / m / 'results.csv'),
                       sha256=sha(cp.OUT / 'evaluation' / m / 'results.csv')) for m in ORDER}
    (AN / 'bitfi_variants_manifest.json').write_text(json.dumps(dict(
        models=ORDER, sources=sources, main_identical_to_original=sources[MAIN]['sha256'] == sha(
            cp.ROOT / '03_result/reevaluation_fullgap_20260929/evaluation/BiTFI-TimesFM3/results.csv'),
        statistics='hour-weighted greenhouse NMAE; equal greenhouse weights; 10,000 greenhouse bootstrap resamples (seed 42); paired two-sided Wilcoxon signed-rank over 10 greenhouses; Holm over the six variant-vs-main comparisons (per scenario and per variable tables: Holm within each scenario/variable)',
        change_definition='change_pct = 100 * (variant / BiTFI-TimesFM3 - 1) of greenhouse-mean NMAE; reduction = -change_pct; main_reduction_vs_variant = 100 * (1 - BiTFI-TimesFM3 / variant)',
        top5_facilities=TOP5), indent=2))
    print(pd.read_csv(AN / 'bitfi_variants_summary.csv').to_string(), flush=True)
    print(pd.read_csv(AN / 'bitfi_variants_paired.csv')[['model', 'change_pct', 'change_low', 'change_high', 'p', 'holm_p', 'greenhouses_variant_worse']].to_string(), flush=True)


if __name__ == '__main__':
    main()

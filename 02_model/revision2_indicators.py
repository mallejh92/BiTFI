"""E5 of revision 2: derived climate indicators and leave-one-greenhouse-out effect sizes.

Every quantity is recomputed from stored gap predictions of the 3,421 main masks
(``evaluation/<MODEL>/predictions.npz`` or ``evaluation/<MODEL>/predictions/<case_id>.npy``,
normalized space, columns ``clean_protocol.COLS``). No model is fitted or run, and all inputs
are only read. Before use, the stored predictions of every analysed variable-case must reproduce
the MAE in ``results.csv`` (tolerance of ``validate_hourly_publication``).

Indicators, compared on the same masks between observed and reconstructed records:
  VPD: cases in which Tin and RH are masked together (Scenarios B and C).
      e_s(T) = 0.6108 exp(17.27 T / (T + 237.3)) kPa  (Tetens equation; FAO-56 Eq. 11)
      VPD    = e_s(T) (1 - RH / 100) kPa              (e_a = e_s RH / 100; FAO-56 Eq. 54, hourly)
      Hourly VPD MAE and the absolute error of daily-mean VPD on complete calendar days.
  Daily outdoor radiation: cases in which Rad is masked (Scenario A single-sensor cases and C).
      H = sum_h Rad_h [W m-2] x 3600 s / 1e6  [MJ m-2 d-1] on complete calendar days.
      Absolute error, per-day relative error 100 |dH| / H_obs, and the ratio 100 sum|dH| / sum H_obs.
  A complete calendar day has all 24 hourly positions (00-23 h, local time) inside the mask.
  Primary results use the unconstrained reconstructions, as in the manuscript. Sensitivity
  variants clip reconstructed RH to [0, 100] % and reconstructed radiation below at 0 W m-2.

Aggregation follows the manuscript: observation-weighted within each greenhouse (hours or
days), equal weight for greenhouses, greenhouse bootstrap (10,000 resamples, seed 42), paired
BiTFI-TimesFM3 reductions with Wilcoxon tests and Holm adjustment over the seven representative
comparisons (``analyze_revision_experiments.summary/site_scores/compare``). BiTFI-TimesFM3-fwd
is an auxiliary comparison outside that family.

The leave-one-greenhouse-out (LOO) analysis repeats the BiTFI-TimesFM3 reductions of
``analysis/all_site_scores.csv`` with each test greenhouse removed in turn.

E3's equal-information regression controls (CovReg-LGBM, CovReg-Ridge) are analysed the same way
when their evaluation folders exist, as a separate two-comparison Holm family (``covreg_holm2``).
They leave the representative outputs and their seven-comparison family unchanged and are written
to ``*_covreg.csv``; their NMAE site scores come from ``evaluation/CovReg-*/results.csv`` because
``all_site_scores.csv`` does not list them.

Run without arguments after E1 has written ``evaluation/_m4_fix_complete.json``.
Outputs: ``<run root>/analysis/revision2/indicators_vpd_*.csv``, ``indicators_radiation_*.csv``,
``loo_effect_sizes*.csv`` and ``indicators_provenance.json``.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / '03_result/revision2_20261019'
READ_ONLY_RUN = ROOT / '03_result/reevaluation_fullgap_20260929'
os.environ.setdefault('BITFI_RUN_ROOT', str(DEFAULT_RUN))
os.environ.setdefault('BITFI_CLEAN', '1')
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(_name, '4')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

import clean_protocol as cp  # noqa: E402
from analyze_revision_experiments import compare, site_scores, summary  # noqa: E402
from preprocessing import VALID_RANGES  # noqa: E402
from revision_config import MAIN_MODELS  # noqa: E402

RUN = cp.OUT.resolve()
FIX_FLAG = RUN / 'evaluation/_m4_fix_complete.json'
REFERENCE = 'BiTFI-TimesFM3'
COMPARATORS = [m for m in MAIN_MODELS if m != REFERENCE]
AUXILIARY = ['BiTFI-TimesFM3-fwd']
MODELS = MAIN_MODELS + AUXILIARY
COVREG = ['CovReg-LGBM', 'CovReg-Ridge']  # E3 equal-information regression controls
MAIN_FAMILIES = [('representative_holm7', COMPARATORS), ('auxiliary', AUXILIARY)]
COVREG_FAMILIES = [('covreg_holm2', COVREG)]
VPD_SCOPES = {'B': ('B',), 'C': ('C',), 'B+C': ('B', 'C')}
RAD_SCOPES = {'A': ('A',), 'C': ('C',), 'A+C': ('A', 'C')}
RH_BOUNDS = VALID_RANGES['RH']            # (0, 100) %, physical screening range of RH
RAD_FLOOR, RAD_CEILING = VALID_RANGES['Rad']  # (0, 2000) W m-2; the sensitivity clips below only
MJ_PER_W_HOUR = 3600. / 1e6
VARIANTS = {'vpd': ('primary', 'clip_RH_0_100'), 'radiation': ('primary', 'clip_Rad_ge_0')}
# metric: (row table, unit, aggregation, paired test). 'weighted' = site_scores with n_eval weights.
VPD_METRICS = {
    'VPD_MAE': ('cases', 'kPa', 'weighted', True),
    'VPD_bias': ('cases', 'kPa', 'weighted', False),
    'VPD_daily_AE': ('days', 'kPa', 'weighted', True),
    'VPD_daily_error': ('days', 'kPa', 'weighted', False),
}
RAD_METRICS = {
    'H_AE': ('days', 'MJ m-2 d-1', 'weighted', True),
    'H_APE': ('days', '%', 'weighted', True),
    'H_rMAE': ('days', '%', 'ratio', True),
    'H_error': ('days', 'MJ m-2 d-1', 'weighted', False),
}


def saturation_vapour_pressure(t):
    """Saturation vapour pressure over water (kPa) at air temperature t (deg C); FAO-56 Eq. 11."""
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def vapour_pressure_deficit(t, rh):
    """Air vapour-pressure deficit (kPa), e_s(T) - e_a with e_a = e_s(T) RH / 100 (FAO-56 Eq. 54)."""
    return saturation_vapour_pressure(t) * (1. - rh / 100.)


def physical(values, scaler):
    """Float64 inverse min-max transform, as evaluate._inverse_transform in clean_metrics.score_fast."""
    return scaler.inverse_transform(np.asarray(values, dtype=np.float64).reshape(-1, 1)).ravel()


def complete_days(index):
    """(date, positions) of calendar days whose 24 hourly positions all lie in the masked interval."""
    dates = index.normalize()
    days = []
    for day in dates.unique():
        positions = np.flatnonzero(dates == day)
        if len(positions) == 24:
            assert np.array_equal(index[positions].hour, np.arange(24))
            days.append((day, positions))
    return days


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prediction_reader(model, manifest):
    """Stored gap predictions (normalized, gap_length_h x 5); packed arrays take precedence."""
    folder = RUN / 'evaluation' / model
    packed = folder / 'predictions.npz'
    if packed.exists():
        bundle = np.load(packed)
        ids, array = bundle['case_id'], bundle['prediction']
        assert np.array_equal(ids, manifest.case_id.to_numpy()), f'{model}: case order differs'
        assert array.shape == (len(manifest), 168, len(cp.COLS)), f'{model}: {array.shape}'
        return (lambda r: array[int(r.case_id), :int(r.gap_length_h)]), 'predictions.npz', sha256(packed)
    directory = folder / 'predictions'
    assert directory.is_dir(), f'{model}: no stored predictions'
    return (lambda r: np.load(directory / f'{int(r.case_id)}.npy')), 'predictions/<case_id>.npy', None


def score_difference(model, r, v, truth, normalized, estimate, saved, scaler):
    """Recomputed case MAE versus results.csv; tolerance of validate_hourly_publication."""
    mae = float(np.abs(estimate - truth).mean())
    quantization = float(np.abs(np.spacing(normalized.astype(np.float32))).astype(np.float64).mean())
    tolerance = 5.0001e-7 + quantization * .5 * float(scaler.data_range_[0]) + 1e-10
    difference = abs(mae - float(saved.MAE))
    if difference > tolerance or int(saved.n_eval) != len(truth):
        raise AssertionError(f'{model} case {r.case_id} {v}: MAE {mae} versus stored {saved.MAE}')
    return difference


def model_records(model, manifest, cases, sites):
    """Case- and day-level indicator records for one configuration (both variants)."""
    read, source, packed_sha = prediction_reader(model, manifest)
    results_path = RUN / 'evaluation' / model / 'results.csv'
    results = pd.read_csv(results_path).query("group_type=='all'")
    assert set(results.model) == {model} and not results.duplicated(['case_id', 'variable']).any()
    results = results.set_index(['case_id', 'variable'])
    digest = hashlib.sha256()
    tables = {k: [] for k in ['vpd_cases', 'vpd_days', 'rad_cases', 'rad_days']}
    worst, checked = 0., 0
    for r in cases.itertuples():
        site = sites[r.greenhouse]
        interval = slice(int(r.start_idx), int(r.end_idx) + 1)
        prediction = np.asarray(read(r))
        assert prediction.shape == (int(r.gap_length_h), len(cp.COLS)), (model, r.case_id, prediction.shape)
        digest.update(np.ascontiguousarray(prediction, dtype=np.float32).tobytes())
        masked = set(r.masked_vars.split(','))
        joint = {'Tin', 'RH'} <= masked
        used = (['Tin', 'RH'] if joint else []) + (['Rad'] if 'Rad' in masked else [])
        truth, estimate = {}, {}
        for v in used:
            j = cp.COLS.index(v)
            truth[v] = physical(site['data'][v].iloc[interval].to_numpy(), site['scaler'][v])
            estimate[v] = physical(prediction[:, j], site['scaler'][v])
            assert np.isfinite(truth[v]).all() and np.isfinite(estimate[v]).all(), (model, r.case_id, v)
            worst = max(worst, score_difference(model, r, v, truth[v], prediction[:, j], estimate[v],
                                                results.loc[(int(r.case_id), v)], site['scaler'][v]))
            checked += 1
        days = complete_days(site['data'].index[interval])
        base = dict(model=model, greenhouse=r.greenhouse, case_id=int(r.case_id), scenario=r.scenario,
                    masked_vars=r.masked_vars, gap_length_h=int(r.gap_length_h), start_time=r.start_time)
        if joint:
            observed = vapour_pressure_deficit(truth['Tin'], truth['RH'])
            rh = estimate['RH']
            for variant, rh_used in zip(VARIANTS['vpd'], [rh, np.clip(rh, *RH_BOUNDS)]):
                reconstructed = vapour_pressure_deficit(estimate['Tin'], rh_used)
                error = reconstructed - observed
                tables['vpd_cases'].append(base | dict(
                    variant=variant, n_eval=len(error), VPD_MAE=float(np.abs(error).mean()),
                    VPD_bias=float(error.mean()), VPD_observed=float(observed.mean()),
                    VPD_reconstructed=float(reconstructed.mean()), RH_above_100_h=int((rh > RH_BOUNDS[1]).sum()),
                    RH_below_0_h=int((rh < RH_BOUNDS[0]).sum()), VPD_negative_h=int((reconstructed < 0).sum()),
                    complete_days=len(days)))
                for day, positions in days:
                    o, e = float(observed[positions].mean()), float(reconstructed[positions].mean())
                    tables['vpd_days'].append(base | dict(
                        variant=variant, date=day.strftime('%Y-%m-%d'), n_eval=1, VPD_daily_observed=o,
                        VPD_daily_reconstructed=e, VPD_daily_error=e - o, VPD_daily_AE=abs(e - o)))
        if 'Rad' in masked:
            rad = estimate['Rad']
            for variant, rad_used in zip(VARIANTS['radiation'], [rad, np.maximum(rad, RAD_FLOOR)]):
                tables['rad_cases'].append(base | dict(
                    variant=variant, n_eval=len(rad), Rad_MAE=float(np.abs(rad_used - truth['Rad']).mean()),
                    Rad_negative_h=int((rad < RAD_FLOOR).sum()), Rad_above_ceiling_h=int((rad > RAD_CEILING).sum()),
                    complete_days=len(days)))
                for day, positions in days:
                    o = float(truth['Rad'][positions].sum() * MJ_PER_W_HOUR)
                    e = float(rad_used[positions].sum() * MJ_PER_W_HOUR)
                    tables['rad_days'].append(base | dict(
                        variant=variant, date=day.strftime('%Y-%m-%d'), n_eval=1, H_observed=o,
                        H_reconstructed=e, H_error=e - o, H_AE=abs(e - o),
                        H_APE=100. * abs(e - o) / o if o > 0 else np.nan,
                        negative_hours=int((rad[positions] < RAD_FLOOR).sum()),
                        observed_zero_hours=int((truth['Rad'][positions] == 0).sum())))
    info = dict(model=model, prediction_source=source, packed_sha256=packed_sha,
                analysed_prediction_sha256=digest.hexdigest(), results_sha256=sha256(results_path),
                variable_cases_checked=checked, max_abs_MAE_difference=worst)
    return {k: pd.DataFrame(v) for k, v in tables.items()}, info


def loo(scores, pairs, metric):
    """Paired reductions 100 (1 - mean model / mean reference) with each greenhouse left out.

    Detail rows also give the left-out greenhouse's own scores and its single-site reduction.
    """
    table = scores.pivot(index='greenhouse', columns='model', values=metric)
    rows, detail = [], []
    for a, b in pairs:
        z = table[[a, b]].dropna()
        values, lower = {}, {}
        site_reduction = 100. * (1. - z[a] / z[b])
        for g in z.index:
            kept = z.drop(index=g)
            values[g] = 100. * (1. - kept[a].mean() / kept[b].mean())
            lower[g] = bool(kept[a].mean() < kept[b].mean())
            detail.append(dict(model=a, reference=b, excluded_greenhouse=g, n_sites=len(kept),
                               model_mean=kept[a].mean(), reference_mean=kept[b].mean(),
                               reduction=values[g], model_lower=lower[g], excluded_model_value=z.at[g, a],
                               excluded_reference_value=z.at[g, b], excluded_site_reduction=site_reduction[g]))
        v = pd.Series(values)
        rows.append(dict(model=a, reference=b, model_mean=z[a].mean(), reference_mean=z[b].mean(),
                         loo_min=v.min(), loo_max=v.max(), loo_min_excluded=v.idxmin(),
                         loo_max_excluded=v.idxmax(), n_loo=len(v), model_lower_all_loo=all(lower.values()),
                         sites_model_lower=int((z[a] < z[b]).sum()), site_reduction_min=site_reduction.min(),
                         site_reduction_max=site_reduction.max()))
    return pd.DataFrame(rows), pd.DataFrame(detail)


def paired(scores, metric, families=MAIN_FAMILIES):
    """BiTFI comparisons with Holm adjustment inside each family.

    Default: representative family (seven comparisons) and the auxiliary forward-only control.
    """
    frames = []
    for family, references in families:
        pairs = [(REFERENCE, m) for m in references]
        tested = compare(scores, pairs, metric)
        ranges, _ = loo(scores, pairs, metric)
        frames.append(tested.merge(ranges, on=['model', 'reference'], validate='one_to_one').assign(family=family))
    return pd.concat(frames, ignore_index=True)


def greenhouse_values(rows, metric, aggregation):
    """Greenhouse scores: observation-weighted means, or 100 sum|dH| / sum H_obs for the ratio form."""
    if aggregation == 'weighted':
        return site_scores(rows, metric)
    z = rows.groupby(['model', 'greenhouse'], observed=True)[['H_AE', 'H_observed']].sum()
    return (100. * z.H_AE / z.H_observed).rename(metric).reset_index()


def analyse(indicator, tables, metrics, scopes, families=MAIN_FAMILIES):
    """Greenhouse scores, bootstrap summaries and paired comparisons for each metric, scope and variant."""
    sites_out, summaries, comparisons = [], [], []
    for metric, (table, unit, aggregation, test) in metrics.items():
        rows = tables[table]
        if aggregation == 'weighted':
            rows = rows[np.isfinite(rows[metric])]  # H_APE is undefined only for days with H_obs = 0
        for scope, scenarios in scopes.items():
            for variant in VARIANTS[indicator]:
                d = rows[rows.scenario.isin(scenarios) & (rows.variant == variant)]
                s = greenhouse_values(d, metric, aggregation)
                counts = d.groupby(['model', 'greenhouse']).n_eval.sum().rename('n_observations').reset_index()
                labels = dict(indicator=indicator, metric=metric, unit=unit, scope=scope, variant=variant)
                sites_out.append(s.merge(counts, on=['model', 'greenhouse']).rename(columns={metric: 'value'}).assign(**labels))
                summaries.append(summary(s.assign(n_eval=1), metric).assign(**labels))
                if test:
                    comparisons.append(paired(s, metric, families).assign(**labels))
    order = ['indicator', 'metric', 'unit', 'scope', 'variant']
    sites_out = pd.concat(sites_out, ignore_index=True)
    summaries = pd.concat(summaries, ignore_index=True)
    comparisons = pd.concat(comparisons, ignore_index=True)
    return (sites_out[order + ['model', 'greenhouse', 'value', 'n_observations']],
            summaries[order + ['model', 'mean', 'sd', 'low', 'high', 'n_sites']],
            comparisons[order + ['family', 'model', 'reference', 'model_mean', 'reference_mean', 'reduction', 'low',
                                 'high', 'p', 'holm_p', 'n_sites', 'loo_min', 'loo_max', 'loo_min_excluded',
                                 'loo_max_excluded', 'n_loo', 'model_lower_all_loo', 'sites_model_lower',
                                 'site_reduction_min', 'site_reduction_max']])


def by_gap(tables, metrics, indicator):
    """Primary-variant summaries by scenario and gap length (descriptive)."""
    frames = []
    for metric, (table, unit, aggregation, _) in metrics.items():
        if aggregation != 'weighted':
            continue
        d = tables[table].query("variant=='primary'")
        d = d[np.isfinite(d[metric])]
        frames.append(summary(d, metric, ['scenario', 'gap_length_h']).assign(indicator=indicator, metric=metric, unit=unit))
    return pd.concat(frames, ignore_index=True)


def greenhouse_mean(rows, column, weight='n_eval'):
    """Equal-greenhouse mean of observation-weighted greenhouse means of a column."""
    z = rows.assign(w=rows[column] * rows[weight]).groupby('greenhouse')[['w', weight]].sum()
    return float((z.w / z[weight]).mean())


def sample_sizes(cases, days, scopes, indicator):
    """Cases, greenhouses, evaluated hours and complete calendar days for each scope and gap length."""
    c = cases.query("model==@REFERENCE and variant=='primary'")
    d = days.query("model==@REFERENCE and variant=='primary'")
    observed = 'VPD_observed' if indicator == 'vpd' else None
    rows = []
    for scope, scenarios in scopes.items():
        for gap in ['all', 6, 12, 24, 72, 168]:
            cc = c[c.scenario.isin(scenarios) & ((c.gap_length_h == gap) if gap != 'all' else True)]
            dd = d[d.scenario.isin(scenarios) & ((d.gap_length_h == gap) if gap != 'all' else True)]
            row = dict(indicator=indicator, scope=scope, gap_length_h=gap, cases=len(cc),
                       greenhouses=cc.greenhouse.nunique(), evaluated_hours=int(cc.n_eval.sum()),
                       cases_with_complete_day=int((cc.complete_days > 0).sum()), complete_days=len(dd),
                       greenhouses_with_complete_day=dd.greenhouse.nunique(),
                       complete_days_per_greenhouse_min=int(dd.groupby('greenhouse').size().min()) if len(dd) else 0,
                       complete_days_per_greenhouse_max=int(dd.groupby('greenhouse').size().max()) if len(dd) else 0)
            if observed and len(cc):
                row['observed_hourly_VPD_kPa'] = greenhouse_mean(cc, observed)
            if indicator == 'vpd' and len(dd):
                row['observed_daily_VPD_kPa'] = greenhouse_mean(dd, 'VPD_daily_observed')
            if indicator == 'radiation' and len(dd):
                row.update(observed_daily_H_MJ=greenhouse_mean(dd, 'H_observed'), observed_daily_H_min=float(dd.H_observed.min()),
                           observed_daily_H_max=float(dd.H_observed.max()), days_H_below_2MJ=int((dd.H_observed < 2).sum()),
                           days_H_zero=int((dd.H_observed <= 0).sum()))
            rows.append(row)
    return pd.DataFrame(rows)


def bounds(cases, days, indicator, scopes):
    """Reconstructions outside the physical screening range, by configuration and scope."""
    rows = []
    for scope, scenarios in scopes.items():
        c = cases[cases.scenario.isin(scenarios) & (cases.variant == 'primary')]
        for model, g in c.groupby('model', sort=False):
            row = dict(indicator=indicator, scope=scope, model=model, evaluated_hours=int(g.n_eval.sum()))
            if indicator == 'vpd':
                row.update(RH_above_100_h=int(g.RH_above_100_h.sum()), RH_below_0_h=int(g.RH_below_0_h.sum()),
                           VPD_negative_h=int(g.VPD_negative_h.sum()))
                row['RH_outside_share'] = (row['RH_above_100_h'] + row['RH_below_0_h']) / row['evaluated_hours']
            else:
                dd = days[(days.model == model) & days.scenario.isin(scenarios) & (days.variant == 'primary')]
                row.update(Rad_negative_h=int(g.Rad_negative_h.sum()), Rad_above_2000_h=int(g.Rad_above_ceiling_h.sum()),
                           complete_day_hours=24 * len(dd), complete_day_negative_h=int(dd.negative_hours.sum()))
                row['Rad_negative_share'] = row['Rad_negative_h'] / row['evaluated_hours']
            rows.append(row)
    return pd.DataFrame(rows)


def main_loo():
    """LOO ranges of the main NMAE reductions from analysis/all_site_scores.csv (verified against results.csv)."""
    path = RUN / 'analysis/all_site_scores.csv'
    scores = pd.read_csv(path)
    frames = [pd.read_csv(RUN / 'evaluation' / m / 'results.csv').query("group_type=='all'") for m in MODELS]
    recomputed = site_scores(pd.concat(frames, ignore_index=True))
    check = recomputed.merge(scores, on=['model', 'greenhouse'], suffixes=('_results', '_file'), validate='one_to_one')
    assert len(check) == 10 * len(MODELS), len(check)
    difference = float((check.NMAE_results - check.NMAE_file).abs().max())
    assert difference < 1e-12, f'all_site_scores.csv differs from evaluation results by {difference}'
    scores = scores[scores.model.isin(MODELS)]
    table = paired(scores, 'NMAE').assign(metric='NMAE', source='analysis/all_site_scores.csv')
    table = table[['metric', 'source', 'family', 'model', 'reference', 'model_mean', 'reference_mean', 'reduction', 'low',
                   'high', 'p', 'holm_p', 'n_sites', 'loo_min', 'loo_max', 'loo_min_excluded', 'loo_max_excluded',
                   'n_loo', 'model_lower_all_loo', 'sites_model_lower', 'site_reduction_min', 'site_reduction_max']]
    _, detail = loo(scores, [(REFERENCE, m) for m in COMPARATORS + AUXILIARY], 'NMAE')
    main_paired = RUN / 'analysis/main_paired.csv'
    agreement = None
    if main_paired.exists():
        stored = pd.read_csv(main_paired).merge(table, on=['model', 'reference'], suffixes=('_stored', ''))
        agreement = float((stored.reduction_stored - stored.reduction).abs().max()) if len(stored) else None
    info = dict(all_site_scores_sha256=sha256(path), max_abs_difference_from_results=difference,
                max_abs_reduction_difference_from_main_paired=agreement)
    return table, detail, info


def covreg_loo():
    """LOO ranges of the NMAE reductions relative to E3's CovReg controls (separate two-comparison family).

    BiTFI site scores come from analysis/all_site_scores.csv. CovReg site scores are recomputed from
    evaluation/CovReg-*/results.csv with the same site_scores rule, because all_site_scores.csv does not
    list them, and are checked against E3's analysis/revision2/covreg_sites.csv when it exists.
    """
    path = RUN / 'analysis/all_site_scores.csv'
    reference = pd.read_csv(path).query('model == @REFERENCE')
    frames = [pd.read_csv(RUN / 'evaluation' / m / 'results.csv').query("group_type=='all'") for m in COVREG]
    for m, d in zip(COVREG, frames):
        assert set(d.model) == {m} and len(d) == 6205 and not d.duplicated(['case_id', 'variable']).any(), m
    scores = pd.concat([reference, site_scores(pd.concat(frames, ignore_index=True))], ignore_index=True)
    assert scores.groupby('model').greenhouse.nunique().eq(10).all() and len(scores) == 10 * (1 + len(COVREG))
    table = paired(scores, 'NMAE', COVREG_FAMILIES).assign(
        metric='NMAE', source='BiTFI: analysis/all_site_scores.csv; CovReg: evaluation/CovReg-*/results.csv')
    table = table[['metric', 'source', 'family', 'model', 'reference', 'model_mean', 'reference_mean', 'reduction', 'low',
                   'high', 'p', 'holm_p', 'n_sites', 'loo_min', 'loo_max', 'loo_min_excluded', 'loo_max_excluded',
                   'n_loo', 'model_lower_all_loo', 'sites_model_lower', 'site_reduction_min', 'site_reduction_max']]
    _, detail = loo(scores, [(REFERENCE, m) for m in COVREG], 'NMAE')
    info = dict(all_site_scores_sha256=sha256(path),
                covreg_results_sha256={m: sha256(RUN / 'evaluation' / m / 'results.csv') for m in COVREG})
    e3_sites = RUN / 'analysis/revision2/covreg_sites.csv'
    if e3_sites.exists():
        stored = pd.read_csv(e3_sites).set_index('greenhouse')[[REFERENCE] + COVREG]
        mine = scores.pivot(index='greenhouse', columns='model', values='NMAE')[[REFERENCE] + COVREG]
        info['max_abs_difference_from_E3_covreg_sites'] = float((stored - mine.loc[stored.index]).abs().max().max())
    e3_paired = RUN / 'analysis/revision2/covreg_paired.csv'
    if e3_paired.exists():
        stored = pd.read_csv(e3_paired).merge(table, on=['model', 'reference'], suffixes=('_stored', ''))
        info['max_abs_reduction_difference_from_E3_covreg_paired'] = float((stored.reduction_stored - stored.reduction).abs().max())
        info['max_abs_holm_p_difference_from_E3_covreg_paired'] = float((stored.holm_p_stored - stored.holm_p).abs().max())
    return table, detail, info


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', type=Path, default=RUN / 'analysis/revision2', help='output directory')
    parser.add_argument('--allow-preliminary', action='store_true',
                        help='run before evaluation/_m4_fix_complete.json exists (development only)')
    args = parser.parse_args()
    if RUN == READ_ONLY_RUN.resolve():
        raise SystemExit('reevaluation_fullgap_20260929 is read-only; set BITFI_RUN_ROOT to the revision-2 root')
    if not FIX_FLAG.exists() and not args.allow_preliminary:
        raise SystemExit(f'{FIX_FLAG} is missing; final indicators require the E1 fallback correction')
    out = args.out.resolve()
    if READ_ONLY_RUN.resolve() in out.parents or out == READ_ONLY_RUN.resolve():
        raise SystemExit('Refusing to write into the read-only full-gap root')
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    manifest = pd.read_csv(RUN / 'mask_manifest.csv')
    assert np.array_equal(manifest.case_id.to_numpy(), np.arange(len(manifest))) and len(manifest) == 3421
    masked = manifest.masked_vars.str.split(',')
    vpd_mask = masked.map(lambda v: {'Tin', 'RH'} <= set(v))
    rad_mask = masked.map(lambda v: 'Rad' in v)
    assert set(manifest[vpd_mask].scenario) == {'B', 'C'} and set(manifest[rad_mask].scenario) == {'A', 'C'}
    assert set(manifest[rad_mask & manifest.scenario.eq('A')].masked_vars) == {'Rad'}
    cases = manifest[vpd_mask | rad_mask]
    sites = {name: cp.site(name) for name in cases.greenhouse.unique()}
    tables = {k: [] for k in ['vpd_cases', 'vpd_days', 'rad_cases', 'rad_days']}
    provenance = []
    for model in MODELS:
        records, info = model_records(model, manifest, cases, sites)
        for k, v in records.items():
            tables[k].append(v)
        provenance.append(info)
        print(model, info['prediction_source'], 'variable-cases', info['variable_cases_checked'],
              'max |MAE - results.csv|', f"{info['max_abs_MAE_difference']:.2e}", flush=True)
    tables = {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}
    for key in ['vpd_cases', 'vpd_days', 'rad_cases', 'rad_days']:
        sizes = tables[key].groupby(['model', 'variant']).size()
        assert sizes.nunique() == 1, f'{key}: unequal record counts across configurations'
    vpd = {'cases': tables['vpd_cases'], 'days': tables['vpd_days']}
    rad = {'cases': tables['rad_cases'], 'days': tables['rad_days']}
    outputs = {}
    for prefix, indicator, data, metrics, scopes in [('indicators_vpd', 'vpd', vpd, VPD_METRICS, VPD_SCOPES),
                                                    ('indicators_radiation', 'radiation', rad, RAD_METRICS, RAD_SCOPES)]:
        sites_out, summaries, comparisons = analyse(indicator, data, metrics, scopes)
        outputs[f'{prefix}_cases.csv'] = data['cases']
        outputs[f'{prefix}_days.csv'] = data['days']
        outputs[f'{prefix}_sites.csv'] = sites_out
        outputs[f'{prefix}_summary.csv'] = summaries
        outputs[f'{prefix}_paired.csv'] = comparisons
        outputs[f'{prefix}_by_gap.csv'] = by_gap(data, metrics, indicator)
        outputs[f'{prefix}_samples.csv'] = sample_sizes(data['cases'], data['days'], scopes, indicator)
        outputs[f'{prefix}_bounds.csv'] = bounds(data['cases'], data['days'], indicator, scopes)
    table, detail, loo_info = main_loo()
    outputs['loo_effect_sizes.csv'] = table
    outputs['loo_effect_sizes_detail.csv'] = detail
    covreg_info = None
    if all((RUN / 'evaluation' / m / 'results.csv').exists() for m in COVREG):
        # E3 controls: same records and statistics, separate Holm family; representative outputs untouched.
        extra = {k: [] for k in tables}
        covreg_sources = []
        for model in COVREG:
            records, info = model_records(model, manifest, cases, sites)
            for k, v in records.items():
                extra[k].append(v)
            covreg_sources.append(info)
            print(model, info['prediction_source'], 'variable-cases', info['variable_cases_checked'],
                  'max |MAE - results.csv|', f"{info['max_abs_MAE_difference']:.2e}", flush=True)
        joint = {k: pd.concat([tables[k][tables[k].model == REFERENCE]] + extra[k], ignore_index=True) for k in tables}
        for key, frame in joint.items():
            assert frame.groupby(['model', 'variant']).size().nunique() == 1, f'{key}: unequal CovReg record counts'
        for prefix, indicator, data, metrics, scopes in [
                ('indicators_vpd', 'vpd', {'cases': joint['vpd_cases'], 'days': joint['vpd_days']}, VPD_METRICS, VPD_SCOPES),
                ('indicators_radiation', 'radiation', {'cases': joint['rad_cases'], 'days': joint['rad_days']}, RAD_METRICS, RAD_SCOPES)]:
            sites_out, summaries, comparisons = analyse(indicator, data, metrics, scopes, COVREG_FAMILIES)
            outputs[f'{prefix}_cases_covreg.csv'] = data['cases'][data['cases'].model != REFERENCE]
            outputs[f'{prefix}_days_covreg.csv'] = data['days'][data['days'].model != REFERENCE]
            outputs[f'{prefix}_sites_covreg.csv'] = sites_out
            outputs[f'{prefix}_summary_covreg.csv'] = summaries
            outputs[f'{prefix}_paired_covreg.csv'] = comparisons
            outputs[f'{prefix}_by_gap_covreg.csv'] = by_gap(data, metrics, indicator)
            outputs[f'{prefix}_bounds_covreg.csv'] = bounds(data['cases'], data['days'], indicator, scopes)
        table, detail, covreg_loo_info = covreg_loo()
        outputs['loo_effect_sizes_covreg.csv'] = table
        outputs['loo_effect_sizes_covreg_detail.csv'] = detail
        covreg_info = dict(models=COVREG, sources=covreg_sources, loo=covreg_loo_info,
                           family='covreg_holm2: Holm over BiTFI vs CovReg-LGBM and BiTFI vs CovReg-Ridge per metric, '
                                  'scope and variant; separate from the seven-comparison representative family',
                           files='*_covreg.csv; BiTFI-TimesFM3 rows in sites/summary/by_gap/bounds are the reference and '
                                 'equal the representative files; cases/days files list only the CovReg rows')
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    flag = None
    if FIX_FLAG.exists():
        text = FIX_FLAG.read_text()
        try:
            content = json.loads(text)
        except json.JSONDecodeError:
            content = text
        flag = dict(path=str(FIX_FLAG.relative_to(RUN)), sha256=sha256(FIX_FLAG), content=content)
    record = dict(
        script='02_model/revision2_indicators.py', run_root=str(RUN), output_directory=str(out),
        created=time.strftime('%Y-%m-%d %H:%M:%S %Z'), elapsed_s=round(time.time() - start, 1),
        e1_fallback_flag=flag, preliminary=flag is None, mask_manifest_sha256=sha256(RUN / 'mask_manifest.csv'),
        models=dict(representative=MAIN_MODELS, reference=REFERENCE, auxiliary=AUXILIARY), sources=provenance,
        formulas=dict(
            saturation_vapour_pressure='e_s(T)=0.6108*exp(17.27*T/(T+237.3)) kPa, T in deg C (Tetens 1930; FAO-56 Eq. 11)',
            vpd='VPD=e_s(T)*(1-RH/100) kPa (e_a=e_s(T)*RH/100, FAO-56 Eq. 54 for hourly data)',
            daily_radiation='H=sum_h Rad_h*3600/1e6 MJ m-2 d-1 over the 24 hourly records of a complete calendar day',
            complete_day='all 24 hourly positions 00-23 h (local timestamps, regular hourly grid) inside the masked interval',
            inverse_scaling='scaler.inverse_transform in float64 for truth (normalized ground truth) and prediction, as clean_metrics.score_fast'),
        variants=dict(primary='unconstrained reconstructions', clip_RH_0_100=f'reconstructed RH clipped to {list(RH_BOUNDS)} %',
                      clip_Rad_ge_0=f'reconstructed Rad clipped below at {RAD_FLOOR} W m-2'),
        aggregation=dict(within_greenhouse='observation-weighted mean (hours for hourly metrics, days for daily metrics); '
                                           'H_rMAE=100*sum|dH|/sum H_obs per greenhouse',
                         across_greenhouses='equal weight', bootstrap='greenhouse resampling, 10,000 draws, numpy default_rng(42)',
                         paired='reduction=100*(1-mean BiTFI/mean reference) over greenhouses with both values; '
                                'two-sided Wilcoxon signed-rank; Holm over the seven representative comparisons per metric, scope and variant '
                                '(CovReg controls: separate two-comparison family covreg_holm2)',
                         loo='each test greenhouse removed in turn; reduction recomputed from the remaining greenhouses'),
        loo=loo_info, covreg=covreg_info, software=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
        files=sorted(outputs))
    (out / 'indicators_provenance.json').write_text(json.dumps(record, indent=2, default=str))
    print('E5 indicators complete', out, f'{time.time() - start:.1f} s', 'preliminary' if flag is None else 'final', flush=True)


if __name__ == '__main__':
    main()

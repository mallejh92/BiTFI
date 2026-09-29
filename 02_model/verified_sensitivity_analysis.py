"""CPU-only diagnostics of saved results; never trains or changes predictions.

Writes only --output. All normalizer choices and exclusions are written to
protocol.json before results are loaded. Re-run against the final revised run
after replacement baseline evaluations have been merged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import TARGET_COLS, VALID_RANGES
from revision_config import MAIN_MODELS

COLS = list(TARGET_COLS.values())
MODELS = [
    "LI", "SeasonalNaive", "Spatial-Ridge", "AG-LightGBM", "AG-RandomForest",
    "AG-DeepAR", "AG-PatchTST", "SAITS", "SAITS-matched-local", "SAITS-spatial",
    "MOMENT", "MOMENT-FT", "Chronos2", "TimesFM2.5", "TimesFM3.0",
    "TimesFM3.0-MV", "TimesFM3.0-COV", "TimesFM3.0-COV-SPA", "BiTFI-TimesFM3",
    "BiTFI-TimesFM3-fwd", "BiTFI-Chronos2",
]
REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n")


def csv(out, name, frame):
    frame.to_csv(out / (name + ".csv"), index=False)


def protocol(root, out):
    p = {
        "version": "verified-sensitivity-20260929-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(root.resolve()), "output_root": str(out.resolve()),
        "execution": "CPU aggregation only; no fitting, inference, or source mutation; clipping is a separately reported sensitivity",
        "normalizers_prespecified": {
            "full_range": "max-min of finite physically screened full-series observations",
            "p01_p99_range": "linear-interpolated quantile(.99)-quantile(.01) of the same observations",
            "scope": "scoring only; both use identical full-series coverage, including withheld observations; no winsorization of errors",
            "zero_policy": "Nonfinite or nonpositive denominators excluded across all models and both normalizers on a common greenhouse-variable support; no epsilon replacement",
            "selection": "Report both definitions, all 21 models and the fixed main eight; do not select a normalizer by model performance",
        },
        "aggregation": "case MAE/denominator weighted by n_eval within each greenhouse, then greenhouses equally; restricted variable summaries follow the same rule",
        "uncertainty": "10000 paired greenhouse bootstrap resamples, seed42, descriptive sensitivity intervals; no new hypothesis-test family",
        "quality": "Only the existing physical-range bounds and 48h exact-constant flag are used; RH100 and CO2 boundary values are descriptive counts, never automatic fault exclusions",
        "prediction_bounds": dict(VALID_RANGES),
        "prediction_bound_tolerance_physical_units": 1e-6,
        "postprocessing_sensitivity": "Separately clip every saved main-eight prediction to the existing common VALID_RANGES in physical units, without altering files or original headline results. Compare original and clipped scores recomputed from the same saved arrays. Report full-range NMAE, P01-P99-normalized MAE, and variable physical-unit MAE. No new bounds, solar masks, or chosen-by-performance rules.",
        "prediction_scope": "Artificially masked genuinely observed target entries only; repeated/overlapping evaluated hours remain occurrences and are not independent observations",
        "missing_runs": "Screened hourly-grid missingness split into leading, trailing, internal and entirely_unobserved; these positions do not establish fault, noninstallation, crop absence, or structural missingness",
        "context": "BiTFI maximum1900 hourly positions on each side; report available positions and genuinely observed target hours separately; no claim filled positions were directly observed",
        "monthly": "Mask counts by starting calendar month; variable-case hours retain original overlap/multiplicity",
        "raw_value_interpretation": "Integer-valued released records do not establish sensor precision, aggregation method, rounding stage, or time zone",
        "manifest_sha256": sha(root / "mask_manifest.csv"),
        "sites_sha256": sha(root / "sites.csv"),
        "script_sha256": sha(__file__),
    }
    dump(out / "protocol.json", p)
    return p


def load_sites(root):
    table = pd.read_csv(root / "sites.csv")
    sites = {}
    for row in table.itertuples():
        with (root / "data" / f"{row.name}.pkl").open("rb") as f:
            obj = pickle.load(f)
        assert obj["name"] == row.name
        sites[row.name] = obj
    return table, sites


def load_results(root, manifest):
    expected = {(int(r.case_id), v) for r in manifest.itertuples() for v in r.masked_vars.split(",")}
    frames, inputs = [], []
    for model in MODELS:
        path = root / "evaluation" / model / "results.csv"
        before = sha(path)
        d = pd.read_csv(path)
        assert sha(path) == before, f"Input changed while reading: {path}"
        d = d[d.group_type.eq("all")].copy()
        assert set(d.model) == {model}
        assert not d.duplicated(["case_id", "variable"]).any()
        assert set(zip(d.case_id, d.variable)) == expected, model
        assert np.isfinite(d[["MAE", "NMAE", "n_eval"]]).all().all()
        assert (d.n_eval > 0).all()
        frames.append(d)
        inputs.append({"model": model, "path": str(path), "sha256": before, "variable_cases": len(d)})
    result = pd.concat(frames, ignore_index=True)
    agreement = result.groupby(["case_id", "variable"]).n_eval.nunique()
    assert agreement.eq(1).all()
    return result, inputs


def site_scores(d, extra=()):
    keys = ["model", "greenhouse", *extra]
    z = d.assign(weighted=d.score * d.n_eval).groupby(keys, observed=True)[["weighted", "n_eval"]].sum()
    return (z.weighted / z.n_eval).rename("score").reset_index()


def summarize(s, extra=()):
    rows = []
    for key, g in s.groupby(["model", *extra], observed=True):
        if not isinstance(key, tuple):
            key = (key,)
        x = g.sort_values("greenhouse").score.to_numpy()
        ix = np.random.default_rng(42).integers(len(x), size=(10000, len(x)))
        lo, hi = np.quantile(x[ix].mean(1), [.025, .975])
        rows.append(dict(zip(["model", *extra], key)) | {
            "mean": x.mean(), "sd": x.std(ddof=1) if len(x) > 1 else np.nan,
            "ci_low": lo, "ci_high": hi, "n_greenhouses": len(x),
        })
    return pd.DataFrame(rows)


def normalizers(root, out, sites, base):
    rows = []
    for name in sorted(base.greenhouse.unique()):
        for v in COLS:
            x = sites[name]["data_raw"][v].dropna().to_numpy(float)
            if len(x):
                low, high = np.quantile(x, [.01, .99], method="linear")
                full = x.max() - x.min()
                robust = high - low
            else:
                low = high = full = robust = np.nan
            rows.append({"greenhouse": name, "variable": v, "n_observations": len(x),
                         "full_range": full, "p01": low, "p99": high,
                         "p01_p99_range": robust,
                         "full_to_robust_ratio": full / robust if robust > 0 else np.nan,
                         "common_valid": bool(np.isfinite(full) and np.isfinite(robust) and full > 0 and robust > 0)})
    norms = pd.DataFrame(rows)
    csv(out, "normalizers", norms)
    z = base.merge(norms, on=["greenhouse", "variable"], validate="many_to_one")
    recomputed = z.MAE / z.full_range
    max_error = float(np.abs(recomputed - z.NMAE).max())
    # Source NMAE is rounded to six decimals; both definitions use saved MAE.
    assert max_error < 5.1e-7, f"Original NMAE cannot be reproduced to saved precision: {max_error}"
    specs = [("full_range_all", "full_range", z),
             ("full_range_common", "full_range", z[z.common_valid]),
             ("p01_p99_common", "p01_p99_range", z[z.common_valid])]
    all_sites, all_summaries, all_variables, effects = [], [], [], []
    for label, denominator, frame in specs:
        f = frame.assign(score=frame.MAE / frame[denominator])
        s = site_scores(f)
        t = summarize(s)
        t["rank_all_21"] = t["mean"].rank(method="min").astype(int)
        t["is_main_eight"] = t.model.isin(MAIN_MODELS)
        t["rank_main_eight"] = np.nan
        selected = t.is_main_eight
        t.loc[selected, "rank_main_eight"] = t.loc[selected, "mean"].rank(method="min")
        all_sites.append(s.assign(normalizer=label))
        all_summaries.append(t.assign(normalizer=label))
        sv = site_scores(f, ["variable"])
        all_variables.append(summarize(sv, ["variable"]).assign(normalizer=label))
        for variable, scores in [("all", s), *[(v, sv[sv.variable.eq(v)]) for v in COLS]]:
            p = scores.pivot(index="greenhouse", columns="model", values="score")
            for comparator in MAIN_MODELS:
                if comparator == "BiTFI-TimesFM3":
                    continue
                pair = p[["BiTFI-TimesFM3", comparator]].dropna()
                x, y = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
                ix = np.random.default_rng(42).integers(len(x), size=(10000, len(x)))
                lo, hi = np.quantile(100 * (1 - x[ix].mean(1) / y[ix].mean(1)), [.025, .975])
                effects.append({"normalizer": label, "variable": variable, "reference": comparator,
                                "bitfi_mean": x.mean(), "reference_mean": y.mean(),
                                "reduction_percent": 100 * (1 - x.mean() / y.mean()),
                                "ci_low": lo, "ci_high": hi, "n_greenhouses": len(x),
                                "greenhouses_bitfi_lower": int((x < y).sum())})
    scores, sums, vars_, eff = map(pd.concat, [all_sites, all_summaries, all_variables, [pd.DataFrame(effects)]])
    csv(out, "normalization_sites", scores)
    csv(out, "normalization_summary", sums.sort_values(["normalizer", "mean"]))
    csv(out, "normalization_variables", vars_)
    csv(out, "normalization_effects", eff)
    excluded = norms[~norms.common_valid]
    report = {"max_original_NMAE_reproduction_error": max_error,
              "saved_NMAE_precision": "six decimals; both scoring definitions recomputed from saved MAE",
              "excluded_greenhouse_variables": excluded[["greenhouse", "variable"]].to_dict("records"),
              "excluded_scored_variable_cases_per_model": int((~z[z.model.eq("BiTFI-TimesFM3")].common_valid).sum()),
              "original_full_range_variable_cases_per_model": int(base.model.eq("BiTFI-TimesFM3").sum()),
              "headline": eff[eff.variable.eq("all") & eff.reference.eq("TimesFM3.0-COV-SPA")].to_dict("records"),
              "bitfi_ranks": sums[sums.model.eq("BiTFI-TimesFM3")].to_dict("records")}
    dump(out / "normalization_checks.json", report)
    return sums, vars_, eff


def runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def data_diagnostics(root, out, table, sites, manifest):
    missing, values, patterns = [], [], []
    for row in table.itertuples():
        raw = sites[row.name]["data_raw"][COLS]
        a = raw.to_numpy(float)
        masks = ~np.isfinite(a)
        for j, v in enumerate(COLS):
            x = a[:, j]
            finite = x[np.isfinite(x)]
            lo, hi = VALID_RANGES[v]
            values.append({"greenhouse": row.name, "group": row.group, "variable": v,
                           "positions": len(x), "n_finite": len(finite),
                           "fraction_missing": masks[:, j].mean(),
                           "integer_fraction_of_observed": np.isclose(finite, np.round(finite), rtol=0, atol=1e-8).mean() if len(finite) else np.nan,
                           "screen_lower_bound": lo, "screen_upper_bound": hi,
                           "n_equal_lower_bound": int((finite == lo).sum()),
                           "n_equal_upper_bound": int((finite == hi).sum()),
                           "n_rh_at_100_descriptive_only": int((finite == 100).sum()) if v == "RH" else 0,
                           "n_co2_at_200_descriptive_only": int((finite == 200).sum()) if v == "CO2" else 0,
                           "n_co2_above_2000_descriptive_only": int((finite > 2000).sum()) if v == "CO2" else 0})
            for start, end in runs(masks[:, j]):
                kind = "entirely_unobserved" if start == 0 and end == len(x) else "leading" if start == 0 else "trailing" if end == len(x) else "internal"
                h = end - start
                bucket = "1-5" if h < 6 else "6-23" if h < 24 else "24-71" if h < 72 else "72-167" if h < 168 else "168-335" if h < 336 else "336-719" if h < 720 else "720+"
                missing.append({"greenhouse": row.name, "group": row.group, "variable": v,
                                "start": str(raw.index[start]), "end": str(raw.index[end - 1]),
                                "hours": h, "position_class": kind, "duration_bin_hours": bucket})
        codes = (masks * (1 << np.arange(len(COLS)))).sum(1)
        for code, n in zip(*np.unique(codes, return_counts=True)):
            label = ",".join(v for j, v in enumerate(COLS) if code & (1 << j)) or "none"
            patterns.append({"greenhouse": row.name, "group": row.group, "missing_variables": label,
                             "hourly_positions": int(n), "fraction_of_site_positions": n / len(raw)})
    miss, val, pat = pd.DataFrame(missing), pd.DataFrame(values), pd.DataFrame(patterns)
    csv(out, "natural_missing_runs", miss)
    csv(out, "raw_value_diagnostics", val)
    csv(out, "missing_cooccurrence_sites", pat)
    csv(out, "missing_cooccurrence_summary", pat.groupby(["group", "missing_variables"], as_index=False).agg(hourly_positions=("hourly_positions", "sum"), n_sites=("greenhouse", "nunique")))
    csv(out, "natural_missing_summary", miss.groupby(["group", "variable", "position_class", "duration_bin_hours"], as_index=False).agg(runs=("hours", "size"), missing_hours=("hours", "sum"), n_sites=("greenhouse", "nunique"), max_hours=("hours", "max")))
    contexts = []
    for r in manifest.itertuples():
        raw = sites[r.greenhouse]["data_raw"]
        l, h = int(r.start_idx), int(r.end_idx) + 1
        left_start, right_end = max(0, l - 1900), min(len(raw), h + 1900)
        for v in r.masked_vars.split(","):
            contexts.append({"case_id": r.case_id, "greenhouse": r.greenhouse, "variable": v,
                             "scenario": r.scenario, "gap_length_h": r.gap_length_h,
                             "left_positions": l - left_start, "right_positions": right_end - h,
                             "left_observed_target_hours": int(raw[v].iloc[left_start:l].notna().sum()),
                             "right_observed_target_hours": int(raw[v].iloc[h:right_end].notna().sum()),
                             "left_shorter_than_maximum": l - left_start < 1900,
                             "right_shorter_than_maximum": right_end - h < 1900})
    ctx = pd.DataFrame(contexts)
    csv(out, "available_context_cases", ctx)
    csv(out, "available_context_summary", ctx.groupby(["scenario", "variable"], as_index=False).agg(
        variable_cases=("case_id", "size"), left_min=("left_positions", "min"), left_median=("left_positions", "median"),
        right_min=("right_positions", "min"), right_median=("right_positions", "median"),
        left_short_cases=("left_shorter_than_maximum", "sum"), right_short_cases=("right_shorter_than_maximum", "sum"),
        left_observed_median=("left_observed_target_hours", "median"), right_observed_median=("right_observed_target_hours", "median")))
    monthly = manifest.copy()
    monthly["start_month"] = pd.to_datetime(monthly.start_time).dt.to_period("M").astype(str)
    monthly["n_variables"] = monthly.masked_vars.str.split(",").str.len()
    monthly["evaluated_variable_hours"] = monthly.gap_length_h * monthly.n_variables
    csv(out, "mask_start_month", monthly.groupby(["start_month", "scenario", "gap_length_h"], as_index=False).agg(
        mask_cases=("case_id", "size"), n_sites=("greenhouse", "nunique"), variable_cases=("n_variables", "sum"),
        evaluated_variable_hours=("evaluated_variable_hours", "sum")))
    overall_context = {}
    one = ctx.drop_duplicates("case_id")
    for scope, d in [("mask_cases", one), ("variable_cases", ctx)]:
        overall_context[scope] = {"n": len(d), "left_short_n": int(d.left_shorter_than_maximum.sum()),
                                  "right_short_n": int(d.right_shorter_than_maximum.sum()),
                                  "both_full_n": int((~d.left_shorter_than_maximum & ~d.right_shorter_than_maximum).sum()),
                                  "left_min": int(d.left_positions.min()), "right_min": int(d.right_positions.min())}
    dump(out / "context_checks.json", overall_context)
    return ctx, miss


def reference_diagnostics(root, out):
    refs = pd.read_csv(root / "quality" / "reference_availability.csv")
    meta = pd.read_excel(REPO / "01_data" / "meta_data.xlsx")
    meta["crop"] = meta.plant.astype(str).str.lower().str.replace("_", " ", regex=False).str.strip()
    assert meta.groupby("name").crop.nunique().max() == 1
    crop = meta.drop_duplicates("name").set_index("name").crop
    slots = refs.assign(reference=refs.references.str.split(",")).explode("reference")
    slots["target_crop"] = slots.greenhouse.map(crop)
    slots["reference_crop"] = slots.reference.map(crop)
    slots["crop_match"] = slots.target_crop.eq(slots.reference_crop)
    assert slots[["target_crop", "reference_crop"]].notna().all().all()
    records = []
    for variable, d in [("all", slots), *list(slots.groupby("variable"))]:
        counts = d.reference.value_counts()
        for rank, (reference, n) in enumerate(counts.items(), 1):
            records.append({"variable": variable, "reference": reference, "reference_crop": crop[reference],
                            "rank": rank, "slots": n, "share": n / len(d)})
    counts = pd.DataFrame(records)
    csv(out, "reference_identity_counts", counts)
    csv(out, "reference_crop_matches", slots.groupby(["variable", "target_crop"], as_index=False).agg(
        slots=("reference", "size"), same_crop_slots=("crop_match", "sum"), same_crop_fraction=("crop_match", "mean")))
    result = {"variable_cases": len(refs), "reference_counts": {str(k): int(v) for k, v in refs.n_references.value_counts().items()},
              "total_slots": len(slots), "top_five_fraction_by_variable": {
                  v: float(g[g["rank"] <= 5].share.sum()) for v, g in counts.groupby("variable")},
              "inference_limit": "Concentration does not identify causation by coverage, crop, distance, or daily/seasonal cycles; no coordinates are available in the supplied metadata"}
    dump(out / "reference_checks.json", result)
    return result


def constant_sensitivity(root, out, base, manifest):
    flags, cases = {}, []
    for name in manifest.greenhouse.unique():
        with np.load(root / "quality" / f"{name}_flags.npz") as z:
            flags[name] = z["constant"].copy()
    for row in manifest.itertuples():
        for v in row.masked_vars.split(","):
            n = int(flags[row.greenhouse][row.start_idx:row.end_idx + 1, COLS.index(v)].sum())
            cases.append({"case_id": row.case_id, "variable": v, "flagged_hours": n})
    cases = pd.DataFrame(cases)
    z = base.merge(cases, on=["case_id", "variable"], validate="many_to_one")
    frames = []
    for label, data in [("all", z), ("exclude_existing_48h_constant_cases", z[z.flagged_hours.eq(0)])]:
        frames.append(summarize(site_scores(data.assign(score=data.NMAE))).assign(subset=label))
    csv(out, "constant_flag_sensitivity", pd.concat(frames, ignore_index=True))
    csv(out, "constant_flagged_cases", cases[cases.flagged_hours.gt(0)])


def prediction_bounds(root, out, sites, manifest, base):
    records, provenance, violation_samples = [], [], {}
    for model in MAIN_MODELS:
        folder = root / "evaluation" / model
        bundle = folder / "predictions.npz"
        arrays = None
        lookup = {}
        if bundle.exists():
            before = sha(bundle)
            with np.load(bundle) as z:
                arrays = z["prediction"].copy()
                lookup = {int(case): i for i, case in enumerate(z["case_id"])}
            assert sha(bundle) == before
            provenance.append({"model": model, "storage": "npz", "sha256": before})
        else:
            provenance.append({"model": model, "storage": "per-case-npy", "path": str(folder / "predictions")})
        saved = base[base.model.eq(model)].set_index(["case_id", "variable"])
        max_mae_error = 0.
        for row in manifest.itertuples():
            prediction = arrays[lookup[int(row.case_id)], :row.gap_length_h] if arrays is not None else np.load(folder / "predictions" / f"{row.case_id}.npy")
            assert prediction.shape == (row.gap_length_h, len(COLS)), (model, row.case_id, prediction.shape)
            obj = sites[row.greenhouse]
            sl = slice(row.start_idx, row.end_idx + 1)
            for v in row.masked_vars.split(","):
                j = COLS.index(v)
                target = obj["data_raw"][v].iloc[sl].to_numpy(float)
                valid = np.isfinite(target)
                scaler = obj["scaler"][v]
                pred = (prediction[:, j].astype(float) - float(scaler.min_[0])) / float(scaler.scale_[0])
                actual_mae = np.abs(pred[valid] - target[valid]).mean()
                expected = saved.loc[(row.case_id, v)]
                assert valid.sum() == int(expected.n_eval)
                max_mae_error = max(max_mae_error, abs(actual_mae - float(expected.MAE)))
                low, high = VALID_RANGES[v]
                finite = np.isfinite(pred[valid])
                below = (pred[valid] < low - 1e-6) & finite
                above = (pred[valid] > high + 1e-6) & finite
                excess = np.maximum(low - pred[valid], pred[valid] - high)
                violation_samples.setdefault((model, v), []).append(excess[below | above])
                records.append({"model": model, "case_id": row.case_id, "greenhouse": row.greenhouse,
                                "variable": v, "n_eval": int(valid.sum()),
                                "below_lower": int(below.sum()), "above_upper": int(above.sum()),
                                "nonfinite": int((~finite).sum()),
                                "zero_truth_occurrences": int((target[valid] == 0).sum()),
                                "below_bound_on_zero_truth": int((below & (target[valid] == 0)).sum()),
                                "below_by_more_than_1": int((pred[valid] < low - 1).sum()),
                                "below_by_more_than_5": int((pred[valid] < low - 5).sum()),
                                "below_by_more_than_10": int((pred[valid] < low - 10).sum()),
                                "minimum_prediction": float(np.nanmin(pred[valid])),
                                "maximum_prediction": float(np.nanmax(pred[valid])),
                                "original_MAE": float(actual_mae),
                                "clipped_MAE": float(np.abs(np.clip(pred[valid], low, high) - target[valid]).mean())})
        assert max_mae_error < 0.002, (model, max_mae_error)
        provenance[-1]["max_MAE_difference_from_saved_scores"] = max_mae_error
        print(f"Bounds checked {model}: maximum MAE reconciliation error {max_mae_error:.8g}", flush=True)
    d = pd.DataFrame(records)
    csv(out, "prediction_bounds_cases", d)
    clipping_sensitivity(out, d)
    sums = d.groupby(["model", "variable"], as_index=False).agg(
        n_eval=("n_eval", "sum"), below_lower=("below_lower", "sum"), above_upper=("above_upper", "sum"),
        nonfinite=("nonfinite", "sum"), minimum_prediction=("minimum_prediction", "min"), maximum_prediction=("maximum_prediction", "max"),
        zero_truth_occurrences=("zero_truth_occurrences", "sum"),
        below_bound_on_zero_truth=("below_bound_on_zero_truth", "sum"),
        below_by_more_than_1=("below_by_more_than_1", "sum"),
        below_by_more_than_5=("below_by_more_than_5", "sum"),
        below_by_more_than_10=("below_by_more_than_10", "sum"))
    sums["pooled_occurrence_violation_fraction"] = (sums.below_lower + sums.above_upper) / sums.n_eval
    g = d.groupby(["model", "variable", "greenhouse"])[["n_eval", "below_lower", "above_upper"]].sum()
    rate = ((g.below_lower + g.above_upper) / g.n_eval).groupby(["model", "variable"]).mean().rename("equal_greenhouse_violation_fraction").reset_index()
    sums = sums.merge(rate, on=["model", "variable"])
    magnitude = []
    for (model, variable), chunks in violation_samples.items():
        x = np.concatenate(chunks)
        magnitude.append({"model": model, "variable": variable,
                          "violation_mean_excess": float(x.mean()) if len(x) else 0.,
                          "violation_median_excess": float(np.median(x)) if len(x) else 0.,
                          "violation_p95_excess": float(np.quantile(x, .95)) if len(x) else 0.,
                          "violation_maximum_excess": float(x.max()) if len(x) else 0.})
    sums = sums.merge(pd.DataFrame(magnitude), on=["model", "variable"])
    csv(out, "prediction_bounds_summary", sums)
    dump(out / "prediction_checks.json", provenance)
    return sums



def clipping_sensitivity(out, cases):
    norms = pd.read_csv(out / "normalizers.csv")
    d = cases.merge(norms, on=["greenhouse", "variable"], validate="many_to_one")
    # Projection onto an interval containing every scored target cannot increase
    # absolute error. This invariant checks both clipping and physical scaling.
    assert (d.clipped_MAE <= d.original_MAE + 1e-9).all()
    summaries, physical, effects = [], [], []
    for output, col in [("original_saved_arrays", "original_MAE"), ("clipped_common_bounds", "clipped_MAE")]:
        physical.append(summarize(site_scores(d.assign(score=d[col]), ["variable"]), ["variable"]).assign(outputs=output))
        for normalizer, denominator in [("full_range_common", "full_range"), ("p01_p99_common", "p01_p99_range")]:
            a = d[d.common_valid].copy()
            a["score"] = a[col] / a[denominator]
            scores = site_scores(a)
            summary = summarize(scores)
            summary["rank_main_eight"] = summary["mean"].rank(method="min").astype(int)
            summaries.append(summary.assign(outputs=output, normalizer=normalizer))
            pair = scores.pivot(index="greenhouse", columns="model", values="score")[["BiTFI-TimesFM3", "TimesFM3.0-COV-SPA"]].dropna()
            x, y = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
            ix = np.random.default_rng(42).integers(len(x), size=(10000, len(x)))
            lo, hi = np.quantile(100 * (1 - x[ix].mean(1) / y[ix].mean(1)), [.025, .975])
            effects.append({"outputs": output, "normalizer": normalizer,
                            "bitfi_mean": x.mean(), "reference_mean": y.mean(),
                            "reduction_percent": 100 * (1 - x.mean() / y.mean()),
                            "ci_low": lo, "ci_high": hi, "n_greenhouses": len(x)})
    csv(out, "clipping_normalization_summary", pd.concat(summaries, ignore_index=True))
    csv(out, "clipping_physical_MAE", pd.concat(physical, ignore_index=True))
    csv(out, "clipping_effects", pd.DataFrame(effects))
    csv(out, "clipping_case_scores", d[["model", "case_id", "greenhouse", "variable", "n_eval", "original_MAE", "clipped_MAE"]])


def write_report(out, p, sums, variables, effects, refs, ctx, miss, bounds):
    headline = effects[effects.variable.eq("all") & effects.reference.eq("TimesFM3.0-COV-SPA")]
    lines = ["# Verified sensitivity analysis", "", f"Input run: `{p['input_root']}`.",
             "No models were fitted or run. Existing predictions, masks and physical screening were retained; physical clipping is evaluated separately and does not replace main results.", "",
             "## Normalization", "",
             "P01–P99 uses the same screened full-series observations as max–min, solely for scoring. Errors are not clipped or winsorized. Both definitions use a common valid greenhouse-variable set.", "",
             "| Normalizer | BiTFI | TimesFM3 local + cross | Reduction | Paired 95% CI |", "|---|---:|---:|---:|---:|"]
    for r in headline.itertuples():
        lines.append(f"| {r.normalizer} | {r.bitfi_mean:.6f} | {r.reference_mean:.6f} | {r.reduction_percent:.2f}% | {r.ci_low:.2f}–{r.ci_high:.2f}% |")
    lines += ["", "All 21 model rankings and the fixed main eight are retained in normalization_summary.csv. Variable results and all main-comparator reductions are in normalization_variables.csv and normalization_effects.csv."]
    rank_table = sums[sums.model.isin(MAIN_MODELS)].pivot(index="model", columns="normalizer", values="rank_main_eight")
    ranks_unchanged = bool(rank_table.full_range_common.eq(rank_table.p01_p99_common).all())
    lines.append(f"The main-eight rank order is unchanged under robust normalization: {ranks_unchanged}.")
    bi_vars = variables[variables.model.eq("BiTFI-TimesFM3")].pivot(index="variable", columns="normalizer", values="mean")
    lines.append(f"CO2 normalized error changes from {bi_vars.loc['CO2', 'full_range_common']:.5f} to {bi_vars.loc['CO2', 'p01_p99_common']:.5f}; RH changes from {bi_vars.loc['RH', 'full_range_common']:.5f} to {bi_vars.loc['RH', 'p01_p99_common']:.5f}. Their ordering reverses, so cross-variable NMAE ranks should not be presented as intrinsic physical reconstruction difficulty.")
    lines += ["", "## Available context and mask placement", ""]
    one = ctx.drop_duplicates("case_id")
    lines.append(f"Of {len(one):,} masks, {int(one.left_shorter_than_maximum.sum()):,} had fewer than 1,900 left positions and {int(one.right_shorter_than_maximum.sum()):,} fewer right positions. Minimum available left/right positions were {one.left_positions.min():,}/{one.right_positions.min():,}. Observed target hours are reported separately from positions; filled or naturally missing positions are not observations.")
    lines += ["", "Mask-start month counts are saved without reweighting or moving masks.", "", "## Natural missingness and data quality", "",
              "Runs are classified as internal, leading, trailing, or entirely unobserved. These labels locate missing records; they cannot determine whether a period was an outage, an uninstalled channel, a non-cropped period, or another structural absence. Counts retain missing hourly-grid positions and physical-screen removals."]
    for v, g in miss.groupby("variable"):
        total = g.hours.sum()
        internal = g[g.position_class.eq("internal")]
        long_share = g[g.hours.ge(168)].hours.sum() / total if total else 0.
        internal_long = internal[internal.hours.ge(168)].hours.sum() / internal.hours.sum() if internal.hours.sum() else 0.
        lines.append(f"- {v}: {int(total):,} missing hourly positions; {100*long_share:.1f}% belong to runs ≥168h; {int(internal.hours.sum()):,} positions occur in internal runs, of which {100*internal_long:.1f}% belong to runs ≥168h. All 34 sites are included; no fault cause is assigned.")
    lines += ["", "Integer-valued records are a property of the released data, not a verified sensor specification. RH100 and CO2 boundary/extreme values were counted but never removed as confirmed faults. Existing 48h constant-flag exclusion is reported separately.", "", "## Reference selection", "",
              f"All {refs['variable_cases']:,} variable-cases used three references. The top five reference facilities supplied {100*refs['top_five_fraction_by_variable']['all']:.2f}% of all slots. Identity/crop-match frequencies do not establish why those sites were selected or identify geography or shared-cycle causation."]
    if bounds is not None:
        lines += ["", "## Prediction physical-range checks", "",
                  "Checks use only the original range-screen bounds. These are broad screening limits, not agronomic tolerances. Counts concern masked observed target occurrences; overlapping evaluations retain multiplicity.", "",
                  "| BiTFI variable | Out-of-range occurrences | Evaluated occurrences | Pooled fraction | Equal-greenhouse fraction |", "|---|---:|---:|---:|---:|"]
        for r in bounds[bounds.model.eq("BiTFI-TimesFM3")].itertuples():
            lines.append(f"| {r.variable} | {r.below_lower+r.above_upper} | {r.n_eval} | {100*r.pooled_occurrence_violation_fraction:.4f}% | {100*r.equal_greenhouse_violation_fraction:.4f}% |")
        radiation = bounds[bounds.model.eq("BiTFI-TimesFM3") & bounds.variable.eq("Rad")].iloc[0]
        lines += ["", f"Negative BiTFI radiation values exceed the lower bound by a median {radiation.violation_median_excess:.2f}, mean {radiation.violation_mean_excess:.2f}, 95th percentile {radiation.violation_p95_excess:.2f}, and maximum {radiation.violation_maximum_excess:.2f} W m−2. {int(radiation.below_bound_on_zero_truth):,} of {int(radiation.below_lower):,} negative values occur where measured irradiance is zero. This is not merely floating-point noise and does not identify astronomical night or a causal fault in directional fusion."]
    lines += ["", "## Interpretation limits", "", "No sensor fault diagnosis, management-event labels, solar-night labels, crop-cycle assumptions or new comparator models were introduced. The sensitivity is descriptive and is not used to select a favorable normalization or modify the test set.", ""]
    clipping_path = out / "clipping_effects.csv"
    if bounds is not None and clipping_path.exists():
        clip = pd.read_csv(clipping_path)
        lines += ["## Separate common-bound clipping sensitivity", "",
                  "Saved predictions were copied in memory and clipped to the original physical-screen bounds. This does not replace the unbounded model outputs or headline results. Original and clipped errors were recomputed from identical saved arrays.", "",
                  "| Outputs | Normalizer | BiTFI | TimesFM3 local + cross | Reduction |", "|---|---|---:|---:|---:|"]
        for r in clip.itertuples():
            lines.append(f"| {r.outputs} | {r.normalizer} | {r.bitfi_mean:.6f} | {r.reference_mean:.6f} | {r.reduction_percent:.2f}% |")
        lines += ["", "Variable physical-unit MAEs are in clipping_physical_MAE.csv; all main-eight NMAE rankings are in clipping_normalization_summary.csv.", ""]
    (out / "summary.md").write_text("\n".join(lines))



def publication_fragments(out):
    """Generate one compact table and short insertion text; no TEX source edits."""
    norm = pd.read_csv(out / "normalization_summary.csv")
    full = norm[norm.normalizer.eq("full_range_common") & norm.model.isin(MAIN_MODELS)].set_index("model")
    robust = norm[norm.normalizer.eq("p01_p99_common") & norm.model.isin(MAIN_MODELS)].set_index("model")
    clip = pd.read_csv(out / "clipping_normalization_summary.csv")
    clipped = clip[clip.outputs.eq("clipped_common_bounds") & clip.normalizer.eq("full_range_common")].set_index("model")
    physical = pd.read_csv(out / "clipping_physical_MAE.csv")
    radiation = physical[physical.variable.eq("Rad")].pivot(index="model", columns="outputs", values="mean")
    labels = {"LI": "Linear interpolation", "SeasonalNaive": "Seasonal naive", "Spatial-Ridge": "Spatial ridge", "AG-LightGBM": "LightGBM", "SAITS-spatial": "SAITS", "MOMENT-FT": "MOMENT", "TimesFM3.0-COV-SPA": "TimesFM3", "BiTFI-TimesFM3": "BiTFI"}
    lines = [
        r"% Generated from verification_20260929/sensitivity CSV files.",
        r"% Regenerate after all revised evaluation results have been merged.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sensitivity to normalization and physical-bound clipping. Parentheses give ranks among the eight configurations, with 1 indicating the lowest error. Clipped NMAE uses the full-series range. Radiation MAE is in W\,m$^{-2}$. SAITS and TimesFM3 use local plus cross-greenhouse inputs; MOMENT is head-tuned and BiTFI uses TimesFM3.}",
        r"\label{tab:verified_sensitivity}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"& \multicolumn{3}{c}{NMAE} & \multicolumn{2}{c}{Radiation MAE} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
        r"Model & Full range & P1--P99 range & Clipped & Original & Clipped \\",
        r"\midrule",
    ]
    for model in full.sort_values("mean", ascending=False).index:
        cells = [labels[model], f"{full.loc[model, 'mean']:.4f} ({int(full.loc[model, 'rank_main_eight'])})", f"{robust.loc[model, 'mean']:.4f} ({int(robust.loc[model, 'rank_main_eight'])})", f"{clipped.loc[model, 'mean']:.4f}", f"{radiation.loc[model, 'original_saved_arrays']:.2f}", f"{radiation.loc[model, 'clipped_common_bounds']:.2f}"]
        if model == "BiTFI-TimesFM3":
            cells = [r"\textbf{" + x + "}" for x in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (out / "supplementary_sensitivity_table.tex").write_text("\n".join(lines))
    effects = pd.read_csv(out / "normalization_effects.csv")
    overall = effects[effects.variable.eq("all") & effects.reference.eq("TimesFM3.0-COV-SPA")].set_index("normalizer")
    ce = pd.read_csv(out / "clipping_effects.csv")
    ce = ce[ce.normalizer.eq("full_range_common")].set_index("outputs")
    bounds = pd.read_csv(out / "prediction_bounds_summary.csv")
    rad = bounds[bounds.model.eq("BiTFI-TimesFM3") & bounds.variable.eq("Rad")].iloc[0]
    r = overall.loc["p01_p99_common"]
    methods = r"We assessed scoring sensitivity by replacing each greenhouse--variable full range with its 1st--99th percentile range, calculated from the same screened observations for scoring only. Both definitions retained the original within-greenhouse hour weighting and equal weighting of greenhouses. All denominators were positive, so the evaluation set was unchanged. Separately, saved predictions from the eight representative configurations were clipped to the common physical-screening bounds, and NMAE and physical-unit MAE were recomputed. The original predictions and primary results were retained."
    discussion = (f"BiTFI retained the lowest NMAE among all 21 configurations under the percentile-range sensitivity; its reduction relative to TimesFM3 with local plus cross-greenhouse inputs was {r.reduction_percent:.2f}\\% (paired 95\\% CI {r.ci_low:.2f}--{r.ci_high:.2f}\\%). The RH--CO$_2$ ordering changed with the denominator, reinforcing that cross-variable NMAE ranks do not directly measure reconstruction difficulty. "
        + f"Negative radiation estimates occurred in {100*rad.pooled_occurrence_violation_fraction:.1f}\\% of evaluated radiation occurrences, of which {100*rad.below_bound_on_zero_truth/rad.below_lower:.1f}\\% had zero measured irradiance. Their median magnitude was {rad.violation_median_excess:.2f}~W\\,m$^{{-2}}$, exceeding numerical round-off. Common-bound clipping reduced BiTFI radiation MAE from {radiation.loc['BiTFI-TimesFM3','original_saved_arrays']:.2f} to {radiation.loc['BiTFI-TimesFM3','clipped_common_bounds']:.2f}~W\\,m$^{{-2}}$, while the overall reduction relative to the same comparator remained {ce.loc['clipped_common_bounds','reduction_percent']:.2f}\\% (Table~\\ref{{tab:verified_sensitivity}}). These results support the aggregate comparison but identify a physical-consistency limitation of the unconstrained outputs.")
    (out / "supplementary_sensitivity_text.tex").write_text("% Supplementary methods insertion\n" + methods + "\n\n% Discussion insertion; optional shortening may retain only the central figures.\n" + discussion + "\n")
    context = json.loads((out / "context_checks.json").read_text())["mask_cases"]
    reference = json.loads((out / "reference_checks.json").read_text())
    notes = ["# Suggested compact insertions", "", "These fragments use the input run named in protocol.json. Regenerate after final baseline replacement. The table contains the requested eight models in descending full-range NMAE order and bolds BiTFI. No main/supplementary manuscript source was edited.", "", "## Methods", "", methods, "", "## Discussion", "", discussion, "", "## Other diagnostics: choose at most one or two sentences", "",
        f"- The 1,900-h context was an upper limit: {context['left_short_n']:,} of {context['n']:,} masks had shorter left contexts and {context['right_short_n']:,} had shorter right contexts; available positions and observed target hours are reported separately in the released diagnostics.",
        f"- All 6,205 variable-cases used three references, with five facilities supplying {100*reference['top_five_fraction_by_variable']['all']:.1f}% of reference slots; this concentration limits conclusions about broadly distributed reference networks.",
        "- Natural missingness included leading and trailing unobserved periods as well as internal gaps, so the duration distribution cannot be interpreted entirely as sensor-outage duration.", "",
        "No daily/seasonal-cycle dominance, sensor-fault cause, crop-cycle interpretation, or device-accuracy claim follows from these diagnostics.", ""]
    (out / "manuscript_insertions.md").write_text("\n".join(notes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-predictions", action="store_true")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    assert out != root and root not in out.parents or "verification_20260929" in out.parts
    out.mkdir(parents=True, exist_ok=True)
    p = protocol(root, out)
    manifest = pd.read_csv(root / "mask_manifest.csv")
    table, sites = load_sites(root)
    base, inputs = load_results(root, manifest)
    dump(out / "result_inputs.json", inputs)
    sums, variables, effects = normalizers(root, out, sites, base)
    print("Normalizer sensitivity aggregated", flush=True)
    ctx, miss = data_diagnostics(root, out, table, sites, manifest)
    refs = reference_diagnostics(root, out)
    constant_sensitivity(root, out, base, manifest)
    bounds = None if args.skip_predictions else prediction_bounds(root, out, sites, manifest, base)
    write_report(out, p, sums, variables, effects, refs, ctx, miss, bounds)
    if bounds is not None:
        publication_fragments(out)
    dump(out / "complete.json", {"input_root": str(root), "completed_utc": datetime.now(timezone.utc).isoformat(),
                                "models": MODELS, "main_models": MAIN_MODELS, "mask_cases": len(manifest),
                                "physical_prediction_checks": not args.skip_predictions,
                                "input_sources_unchanged_after_analysis": all(sha(x["path"]) == x["sha256"] for x in inputs)})
    print("Verified sensitivity complete:", out, flush=True)


if __name__ == "__main__":
    main()

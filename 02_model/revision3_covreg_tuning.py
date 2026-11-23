"""Validation-tuned LightGBM for the equal-information regression control (CEA revision, 2026-11-23).

The revision-2 control CovReg-LGBM used fixed a-priori LightGBM settings.  This script selects
the LightGBM settings and the refinement depth on validation gaps only, then runs the selected
combination once on the test masks.  revision2_covreg.py is imported, not modified: the gap jobs,
two-sided windows, covariate block, masked-input initialization, synchronous passes, fallbacks and
scoring are the same statements as in the untuned control.

Protocol (written to covreg_tuning/protocol.json by --prepare, before any inference):
* Validation gaps: the 1,896 refinement-validation gaps (final 20% of 21 training greenhouses) that
  selected BiTFI's refinement depth.  The target greenhouse is removed from the reference bank.
* Candidates: cfg00 = the revision-2 a-priori settings, plus 39 distinct random draws (seed 42) from
  SPACE; refinement depth 1, 2, 3 or 5 (BiTFI's candidates).  160 candidates in total.
* Validation score: physical MAE / target-greenhouse training-prefix range (validate_refinement_depth),
  hour-weighted within greenhouse, equal greenhouse weights.
* Selection: minimum mean validation NMAE; ties -> fewer passes, then lower configuration id.
* Test: only the selected (configuration, depth) is evaluated on the 3,421 test masks, once.

Usage (run root as in revision 2):
  export BITFI_RUN_ROOT=.../03_result/revision2_20261019 BITFI_CLEAN=1
  .venv/bin/python revision3_covreg_tuning.py --prepare
  .venv/bin/python revision3_covreg_tuning.py --check --workers 40
  .venv/bin/python revision3_covreg_tuning.py --validate --workers 40
  .venv/bin/python revision3_covreg_tuning.py --select
  .venv/bin/python revision3_covreg_tuning.py --test --workers 40
  .venv/bin/python revision3_covreg_tuning.py --analyze
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import clean_protocol as cp
import revision2_covreg as rc

OUT = cp.OUT / "covreg_tuning"
VAL_DIR = OUT / "validation"
TEST_NAME = "CovReg-LGBM-tuned"
TEST_DIR = cp.OUT / "evaluation" / TEST_NAME
VAL_MANIFEST = cp.OUT / "refinement_validation" / "mask_manifest.csv"
CANDIDATE_DEPTHS = (1, 2, 3, 5)
MAX_DEPTH = 5
N_RANDOM = 39
SEED = 42
SPACE = dict(
    n_estimators=[100, 300, 600, 1000],
    learning_rate=[0.02, 0.05, 0.1],
    num_leaves=[7, 15, 31, 63],
    min_child_samples=[5, 10, 20, 50, 100],
    subsample=[0.7, 1.0],
    colsample_bytree=[0.6, 0.8, 1.0],
    reg_lambda=[0.0, 1.0, 10.0],
)
FIXED = dict(reg_alpha=0.0, max_bin=255, random_state=42, deterministic=True, force_row_wise=True,
             n_jobs=1, verbose=-1)
COLS = cp.COLS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_params(draw):
    p = dict(draw)
    p["subsample_freq"] = 1 if p["subsample"] < 1 else 0
    return {**p, **FIXED}


def make_configs():
    base = {k: rc.LGBM_PARAMS[k] for k in SPACE}
    assert all(base[k] in SPACE[k] for k in SPACE), "a-priori settings must lie in the search space"
    rng = np.random.default_rng(SEED)
    draws, seen = [base], {tuple(base[k] for k in SPACE)}
    while len(draws) < N_RANDOM + 1:
        d = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        d = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in d.items()}
        key = tuple(d[k] for k in SPACE)
        if key not in seen:
            seen.add(key)
            draws.append(d)
    return [dict(cfg=i, **d) for i, d in enumerate(draws)]


def load_configs():
    d = pd.read_csv(OUT / "configs.csv")
    out = {}
    for r in d.to_dict("records"):
        draw = {k: (float(r[k]) if isinstance(SPACE[k][0], float) else int(r[k])) for k in SPACE}
        out[int(r["cfg"])] = full_params(draw)
    return out


# ───────────────────────── protocol ─────────────────────────

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise SystemExit("protocol.json exists; the protocol is fixed")
    cfgs = make_configs()
    pd.DataFrame(cfgs).to_csv(OUT / "configs.csv", index=False)
    manifest = pd.read_csv(VAL_MANIFEST)
    protocol = dict(
        purpose="Validation-selected LightGBM settings and refinement depth for the equal-information "
                "regression control; revision-2 CovReg-LGBM used fixed a-priori settings",
        disclosure="The decision to tune was made after the untuned control's test NMAE (0.0372) was known; "
                   "test results do not enter selection",
        validation_manifest=str(VAL_MANIFEST.relative_to(cp.OUT)),
        validation_manifest_sha256=sha256(VAL_MANIFEST),
        validation_cases=int(len(manifest)),
        validation_sites=sorted(manifest.greenhouse.unique().tolist()),
        references="Training greenhouses other than the validation target (target removed from the bank)",
        search=dict(method="random search without replacement", seed=SEED, n_random=N_RANDOM,
                    include_a_priori="cfg00 = revision-2 LGBM_PARAMS", space=SPACE,
                    fixed={**FIXED, "subsample_freq": "1 if subsample < 1 else 0"}),
        depth_candidates=list(CANDIDATE_DEPTHS),
        context=int(rc.make_shell().context_len),
        scoring="Physical MAE divided by per-variable target-greenhouse training-prefix range; "
                "hour-weighted within greenhouse; equal greenhouse weights (validate_refinement_depth)",
        selection="Minimum mean validation NMAE over (configuration, depth); ties -> fewer passes, then lower cfg id",
        test="Only the selected combination is run on the 3,421 test masks, once",
        unchanged=["gap jobs", "two-sided windows (<=1,900 h, >=48 h per side)", "covariate block",
                   "masked-input interpolation initialization", "synchronous passes", "fallback rules",
                   "test scoring (run_clean_evaluation.score)"],
        code_sha256={f: sha256(Path(__file__).resolve().parent / f)
                     for f in ["revision3_covreg_tuning.py", "revision2_covreg.py", "bitfi.py", "spatial.py",
                               "clean_protocol.py", "clean_metrics.py", "run_clean_evaluation.py"]},
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        protocol_created_before_inference=True,
    )
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print(json.dumps({k: v for k, v in protocol.items() if k not in ("validation_sites", "code_sha256")}, indent=2))
    print(pd.DataFrame(cfgs).to_string(index=False))


# ───────────────────────── workers (fork) ─────────────────────────

_G: dict = {}


def _set_params(cfg_id):
    rc.LGBM_PARAMS = _G["configs"][cfg_id]


def _val_work(task):
    cfg_id, pos = task
    row = _G["rows"][pos]
    try:
        _set_params(cfg_id)
        shell = _G["shell"]
        shell.bank.sites = {n: s for n, s in _G["bank"].items() if n != row.greenhouse}
        assert row.greenhouse not in shell.bank.sites
        obj = _G["sites"][row.greenhouse]
        t0 = time.perf_counter()
        _, _, _, states, info, _ = rc.run_case(shell, "CovReg-LGBM", obj, row, MAX_DEPTH)
        spans = _G["spans"][row.greenhouse]
        assert row.start_idx >= obj["cut"]
        out = []
        for step in (0, *CANDIDATE_DEPTHS):
            est = states[step]
            for v in row.masked_vars.split(","):
                ci = COLS.index(v)
                y = obj["data"][v].to_numpy(np.float64)[row.start_idx:row.end_idx + 1]
                p = est[row.start_idx:row.end_idx + 1, ci].astype(np.float64)
                assert np.isfinite(y).all() and np.isfinite(p).all()
                mae = float(np.abs(y - p).mean() * obj["scaler"][v].data_range_[0])
                out.append(dict(cfg=cfg_id, refinements=step, case_id=int(row.case_id), greenhouse=row.greenhouse,
                                scenario=row.scenario, variable=v, gap_length_h=int(row.gap_length_h),
                                n_eval=int(row.gap_length_h), MAE=mae, NMAE=mae / spans[v]))
        fb = sum(1 for x in info for r in x["reasons"] if r != "fit")
        return dict(rows=out, seconds=time.perf_counter() - t0, fallback=fb, error=None)
    except Exception:
        return dict(error=traceback.format_exc(), cfg=cfg_id, case_id=int(row.case_id))


def _test_work(pos):
    row = _G["rows"][pos]
    try:
        _set_params(_G["cfg"])
        obj = _G["sites"][row.greenhouse]
        mr, masked, index, states, info, seconds = rc.run_case(_G["shell"], "CovReg-LGBM", obj, row, MAX_DEPTH)
        rows, windows = rc.score_case(_G["score"], TEST_NAME, _G["shell"], obj, row, mr, masked, index, states)
        return dict(position=pos, case_id=int(row.case_id), rows=rows, windows=windows, info=info,
                    seconds=seconds, error=None)
    except Exception:
        return dict(position=pos, case_id=int(row.case_id), error=traceback.format_exc())


def _pool(workers):
    return mp.get_context("fork").Pool(max(1, int(workers)))


# ───────────────────────── checks ─────────────────────────

def check(workers, n_test=40):
    """(a) cfg00 reproduces the revision-2 test predictions; (b) hidden-target poison invariance."""
    from run_clean_evaluation import score
    configs = load_configs()
    shell = rc.make_shell()
    manifest = pd.read_csv(cp.OUT / "mask_manifest.csv")
    sub = manifest.groupby(["scenario", "gap_length_h"], sort=False).head(3).head(n_test).reset_index(drop=True)
    sites = {n: cp.site(n) for n in sub.greenhouse.unique()}
    _G.update(configs=configs, shell=shell, sites=sites, rows=list(sub.itertuples()), cfg=0, score=score)
    with _pool(workers) as pool:
        res = pool.map(_test_work, range(len(sub)))
    errs = [r["error"] for r in res if r["error"]]
    assert not errs, errs[0]
    ref = np.load(cp.OUT / "evaluation" / "CovReg-LGBM" / "refine5.npz")
    refp = dict(zip(ref["case_id"].tolist(), ref["prediction"]))
    worst = 0.0
    for r in res:
        a = r["windows"][5]
        b = refp[r["case_id"]][:len(a)]
        assert np.array_equal(np.isnan(a), np.isnan(b))
        worst = max(worst, float(np.nanmax(np.abs(a - b))))
    reproduced = worst == 0.0
    print(f"(a) cfg00 vs revision-2 CovReg-LGBM refine5: {len(res)} masks, max |diff| = {worst}", flush=True)

    vman = pd.read_csv(VAL_MANIFEST)
    vsub = vman.groupby("scenario", sort=False).head(1)
    heavy = max(configs, key=lambda k: configs[k]["n_estimators"] * configs[k]["num_leaves"])
    rc.LGBM_PARAMS = configs[heavy]
    bank = dict(shell.bank.sites)
    poison = []
    for row in vsub.itertuples():
        obj = cp.site(row.greenhouse)
        shell.bank.sites = {n: s for n, s in bank.items() if n != row.greenhouse}
        s1 = rc.run_case(shell, "CovReg-LGBM", obj, row, MAX_DEPTH)[3]
        s2 = rc.run_case(shell, "CovReg-LGBM", rc.poisoned(obj, row), row, MAX_DEPTH)[3]
        same = all(np.array_equal(s1[k], s2[k], equal_nan=True) for k in s1)
        poison.append(dict(case_id=int(row.case_id), greenhouse=row.greenhouse, scenario=row.scenario,
                           gap=int(row.gap_length_h), cfg=heavy, invariant=same))
        assert same, f"poison changed validation case {row.case_id}"
    shell.bank.sites = bank
    result = dict(cfg00_reproduces_revision2_test=reproduced, cfg00_max_abs_diff=worst, test_masks_checked=len(res),
                  validation_poison=poison)
    (OUT / "checks.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert reproduced, "cfg00 does not reproduce revision-2 predictions"


# ───────────────────────── validation ─────────────────────────

def validate(workers):
    configs = load_configs()
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(VAL_MANIFEST)
    assert sha256(VAL_MANIFEST) == json.loads((OUT / "protocol.json").read_text())["validation_manifest_sha256"]
    shell = rc.make_shell()
    bank = dict(shell.bank.sites)
    sites = {n: cp.site(n) for n in manifest.greenhouse.unique()}
    spans = {n: {v: float(o["data_raw"][v].iloc[:o["cut"]].max() - o["data_raw"][v].iloc[:o["cut"]].min())
                 for v in COLS} for n, o in sites.items()}
    _G.update(configs=configs, shell=shell, bank=bank, sites=sites, spans=spans, rows=list(manifest.itertuples()))
    todo = [c for c in sorted(configs) if not (VAL_DIR / f"cfg{c:02d}.csv").exists()]
    print(f"validation: {len(manifest)} gaps, {len(todo)} of {len(configs)} configurations to run, {workers} workers",
          flush=True)
    t_all = time.time()
    with _pool(workers) as pool:
        for c in todo:
            t0 = time.time()
            rows, cpu, fb = [], 0.0, 0
            for r in pool.imap_unordered(_val_work, [(c, i) for i in range(len(manifest))], chunksize=4):
                if r["error"]:
                    pool.terminate()
                    raise RuntimeError(f"cfg {c} case {r['case_id']} failed:\n{r['error']}")
                rows.extend(r["rows"])
                cpu += r["seconds"]
                fb += r["fallback"]
            d = pd.DataFrame(rows).sort_values(["case_id", "variable", "refinements"])
            d.to_csv(VAL_DIR / f"cfg{c:02d}.tmp", index=False)
            os.replace(VAL_DIR / f"cfg{c:02d}.tmp", VAL_DIR / f"cfg{c:02d}.csv")
            s = val_summary(d)
            best = s.loc[s.refinements.isin(CANDIDATE_DEPTHS)].sort_values(["mean", "refinements"]).iloc[0]
            meta = dict(cfg=c, wall_s=time.time() - t0, cpu_s=cpu, fallback_job_passes=fb)
            (VAL_DIR / f"cfg{c:02d}.json").write_text(json.dumps(meta))
            print(f"cfg{c:02d} {configs[c]['n_estimators']}x{configs[c]['num_leaves']} lr{configs[c]['learning_rate']} "
                  f"best depth {int(best.refinements)} NMAE {best['mean']:.5f} | {time.time() - t0:.0f}s, "
                  f"fallback {fb}, total {time.time() - t_all:.0f}s", flush=True)


def val_summary(d):
    from analyze_revision_experiments import summary
    s = summary(d.assign(model=d.refinements.astype(str)))
    s["refinements"] = s.model.astype(int)
    return s.drop(columns="model")


def select():
    configs = load_configs()
    rows = []
    for c in sorted(configs):
        f = VAL_DIR / f"cfg{c:02d}.csv"
        if not f.exists():
            raise SystemExit(f"missing {f}")
        s = val_summary(pd.read_csv(f))
        s.insert(0, "cfg", c)
        rows.append(s)
    s = pd.concat(rows, ignore_index=True)
    s.to_csv(OUT / "validation_summary.csv", index=False)
    cand = s[s.refinements.isin(CANDIDATE_DEPTHS)].sort_values(["mean", "refinements", "cfg"])
    chosen = cand.iloc[0]
    a_priori_5 = s.query("cfg==0 and refinements==5").iloc[0]
    result = dict(
        selected_cfg=int(chosen.cfg), selected_refinements=int(chosen.refinements),
        selected_params=configs[int(chosen.cfg)], validation_NMAE=float(chosen["mean"]),
        validation_CI=[float(chosen.low), float(chosen.high)],
        a_priori_cfg00_depth5_validation_NMAE=float(a_priori_5["mean"]),
        bitfi_depth5_validation_NMAE=float(pd.read_csv(cp.OUT / "refinement_validation" / "summary.csv")
                                           .query("refinements==5").iloc[0]["mean"]),
        top10=cand.head(10)[["cfg", "refinements", "mean", "low", "high"]].to_dict("records"),
        candidates=int(len(cand)), test_data_used_for_selection=False,
        protocol_sha256=sha256(OUT / "protocol.json"),
    )
    (OUT / "selection.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


# ───────────────────────── test ─────────────────────────

def test(workers):
    from run_clean_evaluation import score
    sel = json.loads((OUT / "selection.json").read_text())
    if (TEST_DIR / "complete.json").exists():
        raise SystemExit(f"{TEST_DIR}/complete.json exists; the selected combination is evaluated once")
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    configs = load_configs()
    cfg, depth = int(sel["selected_cfg"]), int(sel["selected_refinements"])
    shell = rc.make_shell()
    manifest = pd.read_csv(cp.OUT / "mask_manifest.csv")
    sites = {n: cp.site(n) for n in manifest.greenhouse.unique()}
    _G.update(configs=configs, shell=shell, sites=sites, rows=list(manifest.itertuples()), cfg=cfg, score=score)
    t0 = time.time()
    results, fallbacks = [], Counter()
    print(f"test: cfg{cfg:02d} depth {depth}, {len(manifest)} masks, {workers} workers", flush=True)
    with _pool(workers) as pool:
        for n, r in enumerate(pool.imap_unordered(_test_work, range(len(manifest)), chunksize=2), start=1):
            if r["error"]:
                pool.terminate()
                raise RuntimeError(f"case {r['case_id']} failed:\n{r['error']}")
            results.append(r)
            for x in r["info"]:
                fallbacks.update(v for v in x["reasons"] if v != "fit")
            if n % 500 == 0 or n == len(manifest):
                print(f"test {n}/{len(manifest)} {time.time() - t0:.0f}s fallback {dict(fallbacks)}", flush=True)
    info = rc.write_outputs(TEST_DIR, TEST_NAME, manifest, results, depth)
    cpu = {k: float(sum(r["seconds"][k] for r in results)) for k in range(MAX_DEPTH + 1)}
    summary = dict(
        model=TEST_NAME, regressor="lightgbm.LGBMRegressor", params=configs[cfg], selected_cfg=cfg,
        refinements=depth, results_csv_step=depth, scored_steps=list(rc.SCORED_STEPS),
        tuning="validation-selected (covreg_tuning/selection.json); evaluated once on the test masks",
        selection_sha256=sha256(OUT / "selection.json"), mask_jobs=len(manifest),
        elapsed_s=time.time() - t0, workers=workers, cpu_seconds_per_pass=cpu,
        diagnostics=rc.summarize_info(info), provenance=rc.provenance(),
        **rc.site_nmae(TEST_DIR / "results.csv"),
    )
    (TEST_DIR / "complete.json").write_text(json.dumps(summary, indent=2, default=str))
    print("DONE", json.dumps({k: summary[k] for k in ("model", "selected_cfg", "refinements", "NMAE", "SD", "elapsed_s")}))


# ───────────────────────── analysis ─────────────────────────

def analyze():
    from analyze_revision_experiments import summary, compare, site_scores
    names = {"BiTFI-TimesFM3": "BiTFI", TEST_NAME: "LightGBM (tuned)", "CovReg-LGBM": "LightGBM (a priori)",
             "TimesFM3.0-COV-SPA": "TimesFM3 (local + cross)"}
    frames = []
    for m in names:
        d = pd.read_csv(cp.OUT / "evaluation" / m / "results.csv")
        frames.append(d.assign(model=m))
    d = pd.concat(frames, ignore_index=True)
    allg = d[d.group_type == "all"]
    s = summary(allg)
    pairs = [("BiTFI-TimesFM3", TEST_NAME), (TEST_NAME, "TimesFM3.0-COV-SPA"), (TEST_NAME, "CovReg-LGBM"),
             ("BiTFI-TimesFM3", "TimesFM3.0-COV-SPA")]
    p = compare(site_scores(allg), pairs)
    sites = site_scores(allg).pivot(index="greenhouse", columns="model", values="NMAE")
    p["n_lower"] = [int((sites[a] < sites[b]).sum()) for a, b in pairs]
    strata = {k: summary(allg, extra=[k]) for k in ("scenario", "gap_length_h", "variable")}
    A = cp.OUT / "analysis" / "revision3"
    A.mkdir(parents=True, exist_ok=True)
    s.to_csv(A / "covreg_tuned_summary.csv", index=False)
    p.to_csv(A / "covreg_tuned_paired.csv", index=False)
    sites.to_csv(A / "covreg_tuned_sites.csv")
    for k, t in strata.items():
        t.to_csv(A / f"covreg_tuned_{k}.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.assign(model=s.model.map(names)).round(4).to_string(index=False))
    print(p.round(4).to_string(index=False))
    print(sites.round(4).to_string())
    for k, t in strata.items():
        print(t.pivot(index=k, columns="model", values="mean").rename(columns=names).round(4).to_string())


# ───────────────────────── manuscript tables, indicators and figures ─────────────────────────

A3 = cp.OUT / "analysis" / "revision3"
COVREG_T = ("CovReg-LGBM-tuned", "CovReg-Ridge")      # family (f) after the CEA revision
CONTEXT = ("TimesFM3.0-COV-SPA", "SAITS-spatial", "AG-LightGBM", "Spatial-Ridge")


def tables():
    """Table S7/S11 inputs with the tuned control (revision2_covreg.run_analysis, three controls),
    then family (f) re-tested over the tuned LightGBM and ridge controls only."""
    import analyze_revision_experiments as are
    out = A3 / "covreg"
    rc.MODELS = (*COVREG_T, "CovReg-LGBM")            # untuned control kept as a descriptive row
    rc.run_analysis(out=out)
    load = lambda m: pd.read_csv(cp.OUT / "evaluation" / m / "results.csv").query("group_type=='all'")
    d = pd.concat([load(m) for m in (rc.REFERENCE_MODEL, *COVREG_T, "CovReg-LGBM")], ignore_index=True)
    s = are.site_scores(d)
    fam = are.compare(s, [(rc.REFERENCE_MODEL, m) for m in COVREG_T])
    sites = s.pivot(index="greenhouse", columns="model", values="NMAE")
    fam["n_lower"] = [int((sites[a] < sites[b]).sum()) for a, b in zip(fam.model, fam.reference)]
    fam.to_csv(out / "covreg_paired.csv", index=False)  # family (f): tuned LightGBM and ridge
    are.compare(s, [(rc.REFERENCE_MODEL, "CovReg-LGBM")]).to_csv(out / "covreg_paired_untuned.csv", index=False)
    are.compare(s, [("CovReg-LGBM-tuned", "CovReg-Ridge"), ("CovReg-LGBM-tuned", "CovReg-LGBM")]).to_csv(
        out / "covreg_paired_lgbm_vs_ridge.csv", index=False)
    ctx = pd.concat([d, *[load(m).assign(model=m) for m in CONTEXT]], ignore_index=True)
    are.compare(are.site_scores(ctx), [(m, c) for m in COVREG_T for c in CONTEXT]).to_csv(
        out / "covreg_paired_context.csv", index=False)
    print(fam.round(4).to_string(index=False))


def indicators():
    """revision2_indicators with the tuned LightGBM in family (h); written to analysis/revision3/indicators."""
    import revision2_indicators as ri
    ri.COVREG = list(COVREG_T)
    ri.COVREG_FAMILIES = [("covreg_holm2", ri.COVREG)]

    def covreg_loo_r3():
        # Same computation as ri.covreg_loo; its cross-check points to the revision-3 tables.
        path = ri.RUN / "analysis/all_site_scores.csv"
        reference = pd.read_csv(path).query("model == @ri.REFERENCE")
        frames = [pd.read_csv(ri.RUN / "evaluation" / m / "results.csv").query("group_type=='all'") for m in ri.COVREG]
        scores = pd.concat([reference, ri.site_scores(pd.concat(frames, ignore_index=True))], ignore_index=True)
        table = ri.paired(scores, "NMAE", ri.COVREG_FAMILIES).assign(
            metric="NMAE", source="BiTFI: analysis/all_site_scores.csv; CovReg: evaluation/<model>/results.csv")
        table = table[["metric", "source", "family", "model", "reference", "model_mean", "reference_mean", "reduction",
                       "low", "high", "p", "holm_p", "n_sites", "loo_min", "loo_max", "loo_min_excluded",
                       "loo_max_excluded", "n_loo", "model_lower_all_loo", "sites_model_lower",
                       "site_reduction_min", "site_reduction_max"]]
        _, detail = ri.loo(scores, [(ri.REFERENCE, m) for m in ri.COVREG], "NMAE")
        stored = pd.read_csv(A3 / "covreg/covreg_sites.csv").set_index("greenhouse")[[ri.REFERENCE, *ri.COVREG]]
        mine = scores.pivot(index="greenhouse", columns="model", values="NMAE")[[ri.REFERENCE, *ri.COVREG]]
        info = dict(all_site_scores_sha256=ri.sha256(path),
                    covreg_results_sha256={m: ri.sha256(ri.RUN / "evaluation" / m / "results.csv") for m in ri.COVREG},
                    max_abs_difference_from_revision3_covreg_sites=float((stored - mine.loc[stored.index]).abs().max().max()))
        return table, detail, info

    ri.covreg_loo = covreg_loo_r3
    sys.argv = [sys.argv[0], "--out", str(A3 / "indicators")]
    ri.main()


def _relabel(frame):
    """Figure view: drop the untuned control and give the tuned one the figure scripts' key."""
    f = frame.copy()
    for col in f.columns:
        if f[col].dtype == object:
            if col in ("model", "reference", "base") and f[col].isin(["CovReg-LGBM"]).any():
                f = f[~f[col].eq("CovReg-LGBM") & ~f[col].astype(str).str.startswith("CovReg-LGBM-refine")]
    f = f.rename(columns={"CovReg-LGBM": "_drop"}).drop(columns="_drop", errors="ignore")
    f = f.rename(columns={"CovReg-LGBM-tuned": "CovReg-LGBM"})
    for col in f.columns:
        if f[col].dtype == object:
            f[col] = f[col].str.replace("CovReg-LGBM-tuned", "CovReg-LGBM", regex=False)
    return f


def figure_view():
    """A read-only view of the run root in which analysis/revision2 holds the tuned-control tables."""
    view = OUT / "figure_view"
    if view.exists():
        import shutil
        shutil.rmtree(view)
    view.mkdir(parents=True)
    for p in cp.OUT.iterdir():
        if p.name not in ("analysis", "covreg_tuning"):
            (view / p.name).symlink_to(p.resolve())
    (view / "analysis").mkdir()
    for p in (cp.OUT / "analysis").iterdir():
        if p.name != "revision2":
            (view / "analysis" / p.name).symlink_to(p.resolve())
    r2 = view / "analysis" / "revision2"
    r2.mkdir()
    replaced = {}
    for p in (cp.OUT / "analysis" / "revision2").iterdir():
        covreg_file = p.name.startswith("covreg_") or p.name.endswith("_covreg.csv") or "_covreg_" in p.name
        if not covreg_file:
            (r2 / p.name).symlink_to(p.resolve())
            continue
        src = A3 / ("covreg" if p.name.startswith("covreg_") else "indicators") / p.name
        if src.suffix == ".csv" and src.exists():
            _relabel(pd.read_csv(src)).to_csv(r2 / p.name, index=False)
            replaced[p.name] = str(src.relative_to(cp.OUT))
    (view / "VIEW.json").write_text(json.dumps(dict(
        purpose="Figure inputs with the validation-tuned LightGBM control under the key CovReg-LGBM",
        replaced=replaced, untuned_rows="dropped"), indent=2))
    print(f"view {view}: {len(replaced)} tables replaced")
    return view


def figures():
    """Fig. 6 and Fig. 7 (revision2_figures.render_final) from the figure view."""
    view = figure_view()
    sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))
    import revision2_figures as rf
    import revision2_indicators as rv
    rf.ALIASES["CovReg-LGBM"] = "CovReg-LGBM-tuned"
    # Fig. 7 reads stored predictions through revision2_indicators, bound to the run root and to model
    # folder names: bind it to the view and apply the same alias.
    rv.RUN = view.resolve()
    reader = rv.prediction_reader
    rv.prediction_reader = lambda model, manifest: reader(rf.ALIASES.get(model, model), manifest)
    rf.render_final(argparse.Namespace(run_root=str(view), out=str(OUT / "figures"), font=None,
                                       only=["fig6", "fig7"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for flag in ("prepare", "check", "validate", "select", "test", "analyze", "tables", "indicators", "figures"):
        ap.add_argument(f"--{flag}", action="store_true")
    ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args()
    if a.prepare:
        prepare()
    if a.check:
        check(a.workers)
    if a.validate:
        validate(a.workers)
    if a.select:
        select()
    if a.test:
        test(a.workers)
    if a.analyze:
        analyze()
    if a.tables:
        tables()
    if a.indicators:
        indicators()
    if a.figures:
        figures()

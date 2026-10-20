"""Equal-information covariate-regression controls for BiTFI (revision 2, experiment E3).

CovReg-LGBM and CovReg-Ridge keep every input condition of BiTFI-TimesFM3 and replace
only the frozen backbone with a regression fitted afresh for each missing target of each
mask:

* jobs        - BiTFI's jobs: each maximal unobserved run of a variable that overlaps the
                artificial gap (`_find_gaps`, same statements as `infer_stages`).
* windows     - BiTFI `_sides`: at most the selected context (1,900 h) on each side, the gap
                excluded; a side is used only when it holds >= MIN_SIDE_CTX positions, as in
                the BiTFI refinement passes.
* target      - the actually observed target values inside the used windows (never the
                interpolated natural gaps, never the hidden truth).
* features    - BiTFI `_cov_block` at the same hours: current estimates of the other local
                sensors, four calendar terms and up to three training-greenhouse references
                chosen by `bank.select` from the masked target only.
* schedule    - pass 0 is masked-input linear interpolation; then five synchronous passes,
                each reading the preceding-pass snapshot (BiTFI's validation-selected depth).
                Observed values are never altered.
* fallback    - fewer than MIN_FIT_SAMPLES observed targets, constant features or no usable
                side keep the masked-input interpolation; every fallback is counted.

No TimesFM weights are loaded.  A backbone-free BiTFIImputation shell supplies `_find_gaps`,
`_sides`, `_cov_block` and the reference bank.  `--check-block` (run once with .venv_tfm3 on
CPU) compares the shell's design blocks with the blocks that the production BiTFI-TimesFM3
object builds inside `bitfi_refinement_ablation.infer_stages`.

Usage (run root and environment as in _workspace/revision2/00_plan.md):
  BITFI_RUN_ROOT=.../03_result/revision2_20261019 .venv/bin/python revision2_covreg.py --model CovReg-Ridge --smoke
  BITFI_RUN_ROOT=... .venv/bin/python revision2_covreg.py --model CovReg-LGBM --workers 20
  BITFI_RUN_ROOT=... .venv_tfm3/bin/python revision2_covreg.py --check-block
  BITFI_RUN_ROOT=... .venv/bin/python revision2_covreg.py --analyze
"""
from __future__ import annotations

import os

# Mask-level parallelism with single-threaded numerics; this experiment never uses a GPU.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import copy
import hashlib
import json
import multiprocessing as mp
import platform
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import clean_protocol as cp
from bitfi import MIN_SIDE_CTX, BiTFIImputation, BiTFITimesFM3
from experiment_paths import selected_context
from spatial import DEFAULT_K

MODELS = ("CovReg-LGBM", "CovReg-Ridge")
REFERENCE_MODEL = "BiTFI-TimesFM3"
SCORED_STEPS = (0, 1, 2, 3, 5)
MIN_FIT_SAMPLES = 48
MAX_WORKERS = 20
RIDGE_ALPHA = 1.0
# Fixed a priori; no test-mask or validation tuning.
LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
    subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0,
    max_bin=255, random_state=42, deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1,
)
ORIGINAL_ROOT = cp.ROOT / "03_result/reevaluation_fullgap_20260929"  # read-only
ANALYSIS_DIR = cp.OUT / "analysis" / "revision2"
BLOCK_CHECK_DIR = cp.OUT / "smoke" / "CovReg-block-identity"
CONTEXT_MODELS = ("TimesFM3.0-COV-SPA", "SAITS-spatial", "AG-LightGBM", "Spatial-Ridge",
                  "SeasonalNaive", "MOMENT-FT", "LI")


# ───────────────────────── backbone-free BiTFI shell ─────────────────────────

class CovRegShell(BiTFIImputation):
    """BiTFI without a backbone: inherits `_find_gaps`, `_sides`, `_cov_block` and the bank."""

    def _ready(self) -> bool:
        return True

    def _max_h(self) -> int:
        # The production TimesFM3 job-length cap (no manifest job reaches it).
        return BiTFITimesFM3._max_h(self)


def make_shell() -> CovRegShell:
    """Same constructor arguments as run_clean_evaluation.build('BiTFI-TimesFM3')."""
    selection = json.loads((cp.OUT / "refinement_validation/selection.json").read_text())
    return CovRegShell(bank=cp.CleanNeighborBank(), context_len=selected_context("TimesFM3.0"),
                       k=DEFAULT_K, name="CovReg-shell",
                       refinements=int(selection["selected_refinements"]))


# ───────────────────────── per-mask computation ─────────────────────────

def masked_inputs(shell, obj, row):
    """Per-mask inputs built with the same statements as `infer_stages`."""
    cols = cp.COLS
    mr = cp.masked_case(obj, row)
    masked = mr.masked_data[cols]
    observed = mr.effective_mask[cols].ne(0).values & np.isfinite(masked.values)
    art = mr.artificial_mask[cols].eq(0)
    interp = masked.interpolate(limit_direction="both").ffill().bfill().values.astype(np.float32)
    obs = masked.values.astype(np.float32)
    interp[observed] = obs[observed]
    jobs = []
    for ci, c in enumerate(cols):
        for gs, ge in shell._find_gaps(~observed[:, ci]):
            if art[c].iloc[gs:ge + 1].any():
                jobs.append((ci, gs, ge))
    return mr, masked, observed, obs, interp, jobs


def design(shell, est, index, observed, obs, nb, ci, gs, ge):
    """Regression design of one job from BiTFI's own window and covariate-block code.

    The union [lo, hi) covers the forward block [gs-len(left), ge+1) and the backward block
    [gs, ge+1+len(right)) that BiTFI passes to the backbone.  Training rows are the observed
    target hours of the used windows; prediction rows are the job hours.
    """
    left, right = shell._sides(est[:, ci], gs, ge)
    use_left = len(left) >= MIN_SIDE_CTX
    use_right = bool(shell.bidirectional) and len(right) >= MIN_SIDE_CTX
    if not (use_left or use_right):
        return None
    lo = gs - len(left) if use_left else gs
    hi = ge + 1 + len(right) if use_right else ge + 1
    others = [j for j in range(est.shape[1]) if j != ci]
    names, block = shell._cov_block(est, others, nb, index, lo, hi)
    fit = observed[lo:hi, ci].copy()
    fit[gs - lo:ge + 1 - lo] = False
    return dict(lo=lo, hi=hi, gs=gs, ge=ge, names=names, block=block, fit=fit, y=obs[lo:hi, ci],
                n_left=len(left), n_right=len(right), use_left=use_left, use_right=use_right,
                n_fit_left=int(fit[:gs - lo].sum()), n_fit_right=int(fit[ge + 1 - lo:].sum()))


def regress(model_name, D):
    """Fit on the observed window hours and predict the job hours; None means fallback."""
    X = D["block"].T.astype(np.float64)
    y = D["y"].astype(np.float64)
    gap = slice(D["gs"] - D["lo"], D["ge"] + 1 - D["lo"])
    Xtr, ytr, Xgap = X[D["fit"]], y[D["fit"]], X[gap]
    meta = dict(n_fit=int(len(ytr)), n_features=int(X.shape[1]), n_constant=0)
    if not (np.isfinite(Xtr).all() and np.isfinite(ytr).all() and np.isfinite(Xgap).all()):
        return None, "nonfinite", meta
    if len(ytr) < MIN_FIT_SAMPLES:
        return None, "few_samples", meta
    keep = np.ptp(Xtr, axis=0) > 0
    meta["n_constant"] = int((~keep).sum())
    if not keep.any():
        return None, "constant_features", meta
    Xtr, Xgap = Xtr[:, keep], Xgap[:, keep]
    if model_name == "CovReg-LGBM":
        from lightgbm import LGBMRegressor
        p = LGBMRegressor(**LGBM_PARAMS).fit(Xtr, ytr).predict(Xgap)
    elif model_name == "CovReg-Ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(Xtr)
        p = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(Xtr), ytr).predict(scaler.transform(Xgap))
    else:
        raise ValueError(model_name)
    p = np.asarray(p, np.float32)
    if not np.isfinite(p).all():
        return None, "nonfinite_prediction", meta
    return p, None, meta


def materialize(est, index):
    return pd.DataFrame(est.copy(), index=index, columns=cp.COLS).interpolate(limit_direction="both").ffill().bfill()


def run_case(shell, model_name, obj, row, depth):
    """All passes of one mask: {step: estimate array}, per-job diagnostics, seconds per pass."""
    t0 = time.perf_counter()
    mr, masked, observed, obs, interp, jobs = masked_inputs(shell, obj, row)
    index = masked.index
    refs, info = {}, {}
    for key in jobs:
        ci, gs, ge = key
        nb, picked = (shell.bank.select(cp.COLS[ci], index, obs[:, ci], observed[:, ci], k=shell.k)
                      if shell.use_spatial else (None, []))
        refs[key] = nb
        info[key] = dict(case_id=int(row.case_id), greenhouse=row.greenhouse, scenario=row.scenario,
                         gap_length_h=int(row.gap_length_h), variable=cp.COLS[ci], gs=gs, ge=ge,
                         h_job=ge - gs + 1, n_refs=0 if nb is None else int(nb.shape[0]),
                         refs=";".join(picked), reasons=[])
    seconds = {0: time.perf_counter() - t0}
    est = interp.copy()  # pass 0: masked-input interpolation
    states = {0: est}
    for step in range(1, depth + 1):
        t1 = time.perf_counter()
        out = est.copy()
        for key in jobs:
            ci, gs, ge = key
            if ge - gs + 1 > shell._max_h():
                p, reason, D, meta = None, "too_long", None, {}
            else:
                D = design(shell, est, index, observed, obs, refs[key], ci, gs, ge)
                p, reason, meta = (None, "no_side", {}) if D is None else regress(model_name, D)
            # Synchronous update: every job read the same preceding-pass snapshot `est`.
            out[gs:ge + 1, ci] = interp[gs:ge + 1, ci] if p is None else p
            info[key]["reasons"].append(reason or "fit")
            if step == 1:
                info[key].update(meta)
                if D is not None:
                    info[key].update({k: D[k] for k in ("n_left", "n_right", "use_left", "use_right",
                                                        "n_fit_left", "n_fit_right")})
                    info[key]["covariates"] = ",".join(D["names"])
        est = out
        states[step] = est
        seconds[step] = time.perf_counter() - t1
    return mr, masked, index, states, list(info.values()), seconds


def check_prediction(mr, masked, pred):
    """The same per-case assertions as run_clean_evaluation.infer."""
    art = mr.artificial_mask[cp.COLS].eq(0)
    eff = mr.effective_mask[cp.COLS]
    assert np.isfinite(pred.values[art.values]).all(), "nonfinite prediction"
    np.testing.assert_allclose(pred.values[eff.eq(1).values], masked.values[eff.eq(1).values],
                               rtol=1e-5, atol=1e-6)


def score_case(score, model_name, shell, obj, row, mr, masked, index, states):
    """Score rows and gap-window predictions for the scored passes."""
    rows, windows = {}, {}
    for step, est in states.items():
        pred = materialize(est, index)
        check_prediction(mr, masked, pred)
        windows[step] = pred.iloc[int(row.start_idx):int(row.end_idx) + 1][cp.COLS].to_numpy(np.float32)
        if step in SCORED_STEPS:
            q = score(model_name, shell, obj, row, mr, pred)
            for r in q:
                r.update(case_id=int(row.case_id), refinements=step, protocol=cp.PROTOCOL)
            rows[step] = q
    return rows, windows


# ───────────────────────── worker plumbing (fork) ─────────────────────────

_G: dict = {}


def _work(position):
    row = _G["rows"][position]
    try:
        obj = _G["sites"][row.greenhouse]
        mr, masked, index, states, info, seconds = run_case(_G["shell"], _G["model"], obj, row, _G["depth"])
        rows, windows = score_case(_G["score"], _G["model"], _G["shell"], obj, row, mr, masked, index, states)
        a_same = None
        if row.scenario == "A":
            a_same = all(np.array_equal(states[1], states[k]) for k in range(2, _G["depth"] + 1))
        return dict(position=position, case_id=int(row.case_id), rows=rows, windows=windows,
                    info=info, seconds=seconds, a_same=a_same, error=None)
    except Exception:
        return dict(position=position, case_id=int(row.case_id), error=traceback.format_exc())


def _load_common(model_name):
    from run_clean_evaluation import score
    shell = make_shell()
    manifest = pd.read_csv(cp.OUT / "mask_manifest.csv")
    sites = {n: cp.site(n) for n in manifest.greenhouse.unique()}
    return score, shell, manifest, sites


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    import lightgbm
    import sklearn
    here = Path(__file__).resolve().parent
    files = ["revision2_covreg.py", "bitfi.py", "bitfi_refinement_ablation.py", "spatial.py",
             "clean_protocol.py", "clean_metrics.py", "run_clean_evaluation.py", "analyze_revision_experiments.py"]
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                sklearn=sklearn.__version__, lightgbm=lightgbm.__version__, executable=sys.executable,
                code_sha256={f: sha256(here / f) for f in files}, run_root=str(cp.OUT), protocol=cp.PROTOCOL)


def definition(shell, model_name):
    regressor = (dict(type="lightgbm.LGBMRegressor", params=LGBM_PARAMS) if model_name == "CovReg-LGBM"
                 else dict(type="StandardScaler + sklearn.linear_model.Ridge", alpha=RIDGE_ALPHA, fit_intercept=True))
    return dict(
        model=model_name,
        backbone="none (no TimesFM weights loaded); backbone-free BiTFIImputation shell",
        jobs="BiTFI jobs: maximal unobserved run of each masked variable overlapping the artificial gap",
        windows=f"BiTFI _sides: up to {shell.context_len} h before and after the job, gap excluded; side used "
                f"only if >= MIN_SIDE_CTX={MIN_SIDE_CTX} h (BiTFI refinement rule); bidirectional={shell.bidirectional}",
        target="actually observed target values in the used windows",
        features="BiTFI _cov_block at the same hours: other 4 local sensors (current estimates), calendar "
                 f"hour/day-of-year sin-cos (use_time_covariates={shell.use_time_covariates}), "
                 f"up to k={shell.k} training-greenhouse references (bank.select on masked target)",
        initialization="masked-input linear interpolation (pandas interpolate limit_direction='both', ffill, bfill)",
        refinements=int(shell.refinements),
        update="synchronous; each pass reads the preceding-pass snapshot",
        observed_values="kept",
        fallback=f"masked-input interpolation when < {MIN_FIT_SAMPLES} observed targets, all features constant, "
                 "nonfinite inputs or no side >= MIN_SIDE_CTX; constant individual features dropped",
        job_length_cap=int(shell._max_h()),
        regressor=regressor,
        tuning="none: fixed a priori settings; no test-mask or validation-set tuning",
        scored_steps=list(SCORED_STEPS),
    )


def _counts(series):
    return {str(k): int(v) for k, v in series.value_counts().sort_index().items()}


def _flag(value):
    return isinstance(value, (bool, np.bool_)) and bool(value)


def summarize_info(info):
    d = pd.DataFrame(info)
    reasons = Counter()
    for r in d.itertuples():
        for step, reason in enumerate(r.reasons, start=1):
            if reason != "fit":
                reasons[f"step{step}:{reason}"] += 1
    fallback_jobs = int(d.reasons.map(lambda x: any(v != "fit" for v in x)).sum())
    if "use_left" in d:
        side = d.apply(lambda r: "both" if _flag(r.use_left) and _flag(r.use_right) else (
            "left_only" if _flag(r.use_left) else ("right_only" if _flag(r.use_right) else "none")), axis=1)
    else:
        side = pd.Series(["none"] * len(d))
    return dict(
        variable_cases=int(len(d)),
        fallback_jobs=fallback_jobs,
        fallback_by_step_reason=dict(sorted(reasons.items())),
        sides=_counts(side),
        references=_counts(d.n_refs),
        n_fit=(dict(min=int(d.n_fit.min()), median=float(d.n_fit.median()), max=int(d.n_fit.max()))
               if "n_fit" in d else None),
        constant_features_dropped=_counts(d.n_constant) if "n_constant" in d else None,
        jobs_longer_than_gap=int((d.h_job > d.gap_length_h).sum()),
        max_job_hours=int(d.h_job.max()),
    )


def write_outputs(folder, model_name, manifest, results, depth):
    """Manifest-ordered results.csv/predictions.npz (final pass) and refine{k}.csv/.npz."""
    results = sorted(results, key=lambda r: r["position"])
    case_ids = manifest.case_id.to_numpy()
    assert [r["case_id"] for r in results] == case_ids.tolist()
    for step in SCORED_STEPS:
        rows = [x for r in results for x in r["rows"][step]]
        pred = np.full((len(results), 168, 5), np.nan, np.float32)
        for i, r in enumerate(results):
            pred[i, :len(r["windows"][step])] = r["windows"][step]
        pd.DataFrame(rows).to_csv(folder / f"refine{step}.csv", index=False)
        np.savez_compressed(folder / f"refine{step}.npz", case_id=case_ids, prediction=pred)
        if step == depth:
            pd.DataFrame(rows).to_csv(folder / "results.csv", index=False)
            np.savez_compressed(folder / "predictions.npz", case_id=case_ids, prediction=pred)
    info = [x for r in results for x in r["info"]]
    pd.DataFrame(info).assign(reasons=lambda d: d.reasons.map(",".join)).to_csv(folder / "job_diagnostics.csv", index=False)
    return info


def site_nmae(path):
    d = pd.read_csv(path).query("group_type=='all'")
    s = d.assign(w=d.NMAE * d.n_eval).groupby("greenhouse")[["w", "n_eval"]].sum()
    scores = s.w / s.n_eval
    return dict(rows=len(pd.read_csv(path)), all_cells=len(d), NMAE=float(scores.mean()), SD=float(scores.std()))


# ───────────────────────── smoke: leakage and invariance ─────────────────────────

def edge_case_ids(manifest, sites, shell):
    """First masks with no usable left side, a short right side, and a job longer than its gap."""
    wanted = {"left_lt_min_side": None, "right_lt_min_side": None, "job_longer_than_gap": None}
    for row in manifest.itertuples():
        _, masked, _, _, _, jobs = masked_inputs(shell, sites[row.greenhouse], row)
        n = len(masked)
        for ci, gs, ge in jobs:
            flags = {"left_lt_min_side": min(gs, shell.context_len) < MIN_SIDE_CTX,
                     "right_lt_min_side": min(n - ge - 1, shell.context_len) < MIN_SIDE_CTX,
                     "job_longer_than_gap": ge - gs + 1 > row.gap_length_h}
            for k, v in flags.items():
                if v and wanted[k] is None:
                    wanted[k] = int(row.case_id)
        if all(v is not None for v in wanted.values()):
            break
    return wanted


def poisoned(obj, row):
    """Hidden targets replaced by 1e6 (run_selected_refinement_controls poison check)."""
    q = copy.copy(obj)
    q["data"] = obj["data"].copy()
    q["data_raw"] = obj["data_raw"].copy()
    for v in row.masked_vars.split(","):
        q["data"].iloc[row.start_idx:row.end_idx + 1, q["data"].columns.get_loc(v)] = 1e6
        q["data_raw"].iloc[row.start_idx:row.end_idx + 1, q["data_raw"].columns.get_loc(v)] = 1e6
    return q


def max_diff(a, b):
    both = np.isfinite(a) & np.isfinite(b)
    same_nan = np.array_equal(np.isnan(a), np.isnan(b))
    return float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0, same_nan


def run_smoke(model_name):
    from models.linear_interpolation import LinearInterpolation
    score, shell, manifest, sites = _load_common(model_name)
    depth = int(shell.refinements)
    folder = cp.OUT / "smoke" / model_name
    folder.mkdir(parents=True, exist_ok=True)
    base = manifest.groupby(["scenario", "gap_length_h"], sort=False).head(1)
    edges = edge_case_ids(manifest, sites, shell)
    subset = manifest[manifest.case_id.isin(set(base.case_id) | {v for v in edges.values() if v is not None})]
    checks, results = [], []
    t0 = time.time()
    for position, row in enumerate(subset.itertuples()):
        obj = sites[row.greenhouse]
        tc = time.perf_counter()
        mr, masked, index, states, info, seconds = run_case(shell, model_name, obj, row, depth)
        case_seconds = time.perf_counter() - tc
        rows, windows = score_case(score, model_name, shell, obj, row, mr, masked, index, states)
        # Hidden truth poisoned with 1e6: every pass must be unchanged.
        _, _, _, p_states, _, _ = run_case(shell, model_name, poisoned(obj, row), row, depth)
        _, _, _, r_states, _, _ = run_case(shell, model_name, obj, row, depth)
        poison = {k: max_diff(states[k], p_states[k]) for k in states}
        repeat = {k: max_diff(states[k], r_states[k]) for k in states}
        li = LinearInterpolation().impute(mr.masked_data[cp.COLS], mr.effective_mask[cp.COLS])
        li_diff, _ = max_diff(materialize(states[0], index).values, li.values.astype(np.float64))
        poison_exact = all(np.array_equal(states[k], p_states[k], equal_nan=True) for k in states)
        repeat_exact = all(np.array_equal(states[k], r_states[k], equal_nan=True) for k in states)
        a_same = (all(np.array_equal(states[1], states[k]) for k in range(2, depth + 1))
                  if row.scenario == "A" else None)
        changes = {k: float(np.nanmax(np.abs(states[k] - states[k - 1]))) for k in range(1, depth + 1)}
        assert poison_exact, f"hidden-target poison changed case {row.case_id}: {poison}"
        assert repeat_exact, f"non-deterministic case {row.case_id}: {repeat}"
        assert li_diff < 1e-6, f"pass 0 differs from LI by {li_diff}"
        if a_same is not None:
            assert a_same, f"scenario A passes differ for case {row.case_id}"
        tags = [k for k, v in edges.items() if v == int(row.case_id)]
        checks.append(dict(case_id=int(row.case_id), greenhouse=row.greenhouse, scenario=row.scenario,
                           gap=int(row.gap_length_h), masked_vars=row.masked_vars, edge=tags,
                           hidden_target_poison_invariant=poison_exact,
                           poison_max_abs_diff=max(v[0] for v in poison.values()),
                           repeat_deterministic=repeat_exact, pass0_equals_LI_max_abs_diff=li_diff,
                           scenarioA_passes_identical=a_same, max_change_per_pass=changes,
                           fallback=[x["reasons"] for x in info], n_refs=[x["n_refs"] for x in info],
                           n_fit=[x.get("n_fit") for x in info], seconds=case_seconds,
                           seconds_per_pass=seconds))
        results.append(dict(position=position, case_id=int(row.case_id), rows=rows, windows=windows, info=info))
        print(model_name, "smoke", row.case_id, row.scenario, row.gap_length_h, f"{case_seconds:.2f}s",
              "fallback", [x["reasons"] for x in info if any(v != "fit" for v in x["reasons"])], flush=True)
    # A-B-A execution order (no state is shared between masks).
    rs = list(subset.itertuples())
    a, b = rs[0], rs[-1]
    first = run_case(shell, model_name, sites[a.greenhouse], a, depth)[3]
    run_case(shell, model_name, sites[b.greenhouse], b, depth)
    again = run_case(shell, model_name, sites[a.greenhouse], a, depth)[3]
    aba = all(np.array_equal(first[k], again[k], equal_nan=True) for k in first)
    assert aba
    info = write_outputs(folder, model_name, subset.reset_index(drop=True), results, depth)
    invariance = dict(model=model_name, cases=len(subset), edge_cases=edges,
                      hidden_target_poison_invariant=all(c["hidden_target_poison_invariant"] for c in checks),
                      poison_rule="hidden targets set to 1e6 in data and data_raw over the artificial gap",
                      repeat_deterministic=all(c["repeat_deterministic"] for c in checks),
                      ABA_order_invariance=aba,
                      pass0_equals_LI=max(c["pass0_equals_LI_max_abs_diff"] for c in checks),
                      scenarioA_passes_identical=all(c["scenarioA_passes_identical"] for c in checks
                                                     if c["scenarioA_passes_identical"] is not None),
                      checks=checks)
    (folder / "invariance.json").write_text(json.dumps(invariance, indent=2))
    summary = dict(definition(shell, model_name), mode="smoke", cases=len(subset), elapsed_s=time.time() - t0,
                   diagnostics=summarize_info(info), provenance=provenance(), **site_nmae(folder / "results.csv"))
    (folder / "complete.json").write_text(json.dumps(summary, indent=2, default=str))
    print("SMOKE DONE", json.dumps({k: v for k, v in invariance.items() if k != "checks"}), flush=True)


# ───────────────────────── full evaluation ─────────────────────────

def run_full(model_name, workers, overwrite=False):
    folder = cp.OUT / "evaluation" / model_name
    if (folder / "complete.json").exists() and not overwrite:
        raise SystemExit(f"{folder}/complete.json exists; pass --overwrite to recompute")
    folder.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(int(workers), MAX_WORKERS))
    score, shell, manifest, sites = _load_common(model_name)
    depth = int(shell.refinements)
    _G.update(rows=list(manifest.itertuples()), sites=sites, shell=shell, model=model_name,
              depth=depth, score=score)
    t0 = time.time()
    results, fallbacks = [], Counter()
    print(f"{model_name}: {len(manifest)} masks, {workers} workers, depth {depth}, context {shell.context_len}",
          flush=True)
    with mp.get_context("fork").Pool(workers) as pool:
        for n, res in enumerate(pool.imap_unordered(_work, range(len(manifest)), chunksize=2), start=1):
            if res["error"]:
                pool.terminate()
                raise RuntimeError(f"case {res['case_id']} failed:\n{res['error']}")
            results.append(res)
            for x in res["info"]:
                for v in x["reasons"]:
                    if v != "fit":
                        fallbacks[v] += 1
            if n % 100 == 0 or n == len(manifest):
                el = time.time() - t0
                print(f"{model_name} {n}/{len(manifest)} masks {el:.0f}s elapsed, ETA {el / n * (len(manifest) - n):.0f}s, "
                      f"fallback job-passes {dict(fallbacks)}", flush=True)
    elapsed = time.time() - t0
    info = write_outputs(folder, model_name, manifest, results, depth)
    a_flags = [r["a_same"] for r in results if r["a_same"] is not None]
    cpu = {k: float(sum(r["seconds"][k] for r in results)) for k in range(depth + 1)}
    smoke = cp.OUT / "smoke" / model_name / "invariance.json"
    block = BLOCK_CHECK_DIR / "block_identity.json"
    summary = dict(
        definition(shell, model_name), mode="evaluation", mask_jobs=len(manifest), elapsed_s=elapsed,
        workers=workers, cpu_seconds_per_pass=cpu, cpu_seconds_total=float(sum(cpu.values())),
        scenarioA_passes_identical=dict(cases=len(a_flags), identical=int(sum(a_flags))),
        diagnostics=summarize_info(info),
        smoke=({k: v for k, v in json.loads(smoke.read_text()).items() if k != "checks"} if smoke.exists() else None),
        block_identity=({k: v for k, v in json.loads(block.read_text()).items() if k not in ("records", "config")}
                        if block.exists() else None),
        provenance=provenance(), **site_nmae(folder / "results.csv"),
    )
    (folder / "complete.json").write_text(json.dumps(summary, indent=2, default=str))
    print("DONE", json.dumps({k: summary[k] for k in ("model", "NMAE", "SD", "elapsed_s", "cpu_seconds_total")}),
          flush=True)


# ───────────────────────── covariate-block identity (.venv_tfm3, CPU) ─────────────────────────

def run_block_check(depth_check=2, threads=16):
    """Compare the shell's design blocks with the production BiTFI object's `_cov_block` calls.

    The production object is built by run_clean_evaluation.build('BiTFI-TimesFM3'); its
    CheckpointManager is redirected to a temporary directory so that the shared run root is
    not touched.  TimesFM3 runs on CPU; `torch.cuda.synchronize` (timing only) is a no-op.
    """
    import torch
    torch.set_num_threads(threads)
    torch.cuda.synchronize = lambda *a, **k: None
    import bitfi_refinement_ablation as bra
    import run_clean_evaluation as rce
    from checkpoint import CheckpointManager
    BLOCK_CHECK_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(cp.OUT / "mask_manifest.csv")
    sites = {n: cp.site(n) for n in manifest.greenhouse.unique()}
    shell = make_shell()
    base = manifest.groupby(["scenario", "gap_length_h"], sort=False).head(1)
    edges = edge_case_ids(manifest, sites, shell)
    extra = {v for v in edges.values() if v is not None}
    extra |= set(manifest[(manifest.greenhouse == "PF_0025102_01") & (manifest.scenario == "C")
                          & (manifest.gap_length_h == 168)].case_id.head(1))
    subset = manifest[manifest.case_id.isin(set(base.case_id) | extra)]
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        rce.CheckpointManager = lambda root: CheckpointManager(Path(tmp) / "models")
        real = rce.build(REFERENCE_MODEL, selected_context("TimesFM3.0"))
    torch.set_float32_matmul_precision("highest")
    load_s = time.time() - t0
    config = {}
    for attr in ("context_len", "k", "use_spatial", "use_time_covariates", "bidirectional", "refinements"):
        config[attr] = dict(bitfi=getattr(real, attr), shell=getattr(shell, attr),
                            equal=getattr(real, attr) == getattr(shell, attr))
    config["max_h"] = dict(bitfi=real._max_h(), shell=shell._max_h(), equal=real._max_h() == shell._max_h())
    config["use_local"] = dict(bitfi=getattr(real, "use_local", True), shell=True,
                               equal=getattr(real, "use_local", True) is True)
    bank_equal = (set(real.bank.sites) == set(shell.bank.sites) and real.bank.target_vars == shell.bank.target_vars
                  and all(real.bank.sites[n].equals(shell.bank.sites[n]) for n in shell.bank.sites))
    config["bank"] = dict(bitfi_sites=len(real.bank.sites), shell_sites=len(shell.bank.sites), equal=bank_equal)
    records, current = [], {"step": 0}
    original = real._cov_block

    def recording(est, others, nb, index, lo, hi):
        names, block = original(est, others, nb, index, lo, hi)
        records.append(dict(step=current["step"], est=est.copy(), others=list(others),
                            nb=None if nb is None else nb.copy(), lo=int(lo), hi=int(hi),
                            names=list(names), block=block.copy()))
        return names, block

    real._cov_block = recording
    examples = [(sites[r.greenhouse], r) for r in subset.itertuples()]
    snapshots = {}
    t1 = time.time()
    gen = bra.infer_stages(real, examples, max_refinements=depth_check)
    for step in range(depth_check + 1):
        current["step"] = step
        k, batch, _ = next(gen)
        assert k == step
        snapshots[step] = [pred.values.copy() for _, pred in batch]
    infer_s = time.time() - t1
    # Expected call order in each refinement pass: examples, then jobs, forward then backward.
    expected = []
    for j, (obj, row) in enumerate(examples):
        mr, masked, observed, obs, interp, jobs = masked_inputs(shell, obj, row)
        n = len(masked)
        for ci, gs, ge in jobs:
            nb, picked = shell.bank.select(cp.COLS[ci], masked.index, obs[:, ci], observed[:, ci], k=shell.k)
            left, right = min(gs, shell.context_len), min(n - ge - 1, shell.context_len)
            item = dict(j=j, case_id=int(row.case_id), scenario=row.scenario, variable=cp.COLS[ci], ci=ci,
                        gs=gs, ge=ge, index=masked.index, observed=observed, obs=obs, nb=nb, refs=picked)
            if left >= MIN_SIDE_CTX:
                expected.append(dict(item, direction="f", lo=gs - left, hi=ge + 1))
            if real.bidirectional and right >= MIN_SIDE_CTX:
                expected.append(dict(item, direction="b", lo=gs, hi=ge + 1 + right))
    rows, worst = [], 0.0
    for step in range(1, depth_check + 1):
        got = [r for r in records if r["step"] == step]
        assert len(got) == len(expected), (step, len(got), len(expected))
        for e, r in zip(expected, got):
            D = design(shell, r["est"], e["index"], e["observed"], e["obs"], e["nb"], e["ci"], e["gs"], e["ge"])
            if e["direction"] == "f":
                mine = D["block"][:, :e["ge"] + 1 - D["lo"]]
                window_ok = D["lo"] == r["lo"] and r["hi"] == e["ge"] + 1
            else:
                mine = D["block"][:, e["gs"] - D["lo"]:]
                window_ok = D["hi"] == r["hi"] and r["lo"] == e["gs"]
            block_equal = mine.shape == r["block"].shape and np.array_equal(mine, r["block"])
            diff = float(np.max(np.abs(mine - r["block"]))) if mine.shape == r["block"].shape else float("inf")
            worst = max(worst, diff)
            nb_equal = ((r["nb"] is None and e["nb"] is None) or
                        (r["nb"] is not None and e["nb"] is not None and np.array_equal(r["nb"], e["nb"])))
            snapshot_equal = np.array_equal(r["est"], snapshots[step - 1][e["j"]].astype(r["est"].dtype), equal_nan=True)
            # Training rows lie inside BiTFI's own context windows (forward: left, backward: right).
            train_pos = np.flatnonzero(D["fit"]) + D["lo"]
            inside = bool(np.all(((train_pos >= e["gs"] - D["n_left"]) & (train_pos < e["gs"]))
                                 | ((train_pos > e["ge"]) & (train_pos <= e["ge"] + D["n_right"]))))
            rows.append(dict(step=step, case_id=e["case_id"], scenario=e["scenario"], variable=e["variable"],
                             direction=e["direction"], lo=r["lo"], hi=r["hi"], names=",".join(r["names"]),
                             names_equal=r["names"] == D["names"], window_equal=bool(window_ok),
                             block_equal=bool(block_equal), max_abs_diff=diff, references_equal=bool(nb_equal),
                             refs=";".join(e["refs"]), snapshot_equals_materialized=bool(snapshot_equal),
                             training_rows_inside_bitfi_windows=inside, n_training_rows=int(len(train_pos))))
    table = pd.DataFrame(rows)
    table.to_csv(BLOCK_CHECK_DIR / "block_identity_records.csv", index=False)
    ok = bool(table[["names_equal", "window_equal", "block_equal", "references_equal",
                     "snapshot_equals_materialized", "training_rows_inside_bitfi_windows"]].all().all())
    result = dict(
        reference_object="run_clean_evaluation.build('BiTFI-TimesFM3', selected_context('TimesFM3.0')) on CPU, "
                         "CheckpointManager redirected to a temporary directory",
        procedure=f"bitfi_refinement_ablation.infer_stages(max_refinements={depth_check}) with real TimesFM3 "
                  "forecasts; every _cov_block call recorded with its est snapshot; shell design() rebuilt "
                  "from the recorded snapshot and the shell's own bank.select",
        cases=subset.case_id.astype(int).tolist(), edge_cases=edges,
        variable_cases=int(len({(e["case_id"], e["variable"]) for e in expected})),
        records_checked=int(len(table)), all_identical=ok and all(v["equal"] for v in config.values()),
        max_abs_diff=worst, config_equal=all(v["equal"] for v in config.values()),
        covariate_names=sorted(table.names.unique().tolist()),
        load_seconds=load_s, inference_seconds=infer_s, torch_threads=threads, config=config,
        provenance=dict(python=platform.python_version(), executable=sys.executable, torch=str(torch.__version__),
                        code_sha256={f: sha256(Path(__file__).resolve().parent / f) for f in (
                            "revision2_covreg.py", "bitfi.py", "bitfi_refinement_ablation.py", "spatial.py",
                            "clean_protocol.py", "run_clean_evaluation.py")}),
    )
    (BLOCK_CHECK_DIR / "block_identity.json").write_text(json.dumps(result, indent=2, default=str))
    print("BLOCK CHECK", json.dumps({k: v for k, v in result.items() if k != "config"}, default=str), flush=True)
    assert result["all_identical"], "covariate block mismatch"


# ───────────────────────── first summary ─────────────────────────

def run_analysis(eval_root=None, out=None, cases=None):
    """First summary; `cases` restricts every input to a subset (used only to test this code)."""
    import analyze_revision_experiments as are
    eval_root = Path(eval_root) if eval_root else cp.OUT / "evaluation"
    out = Path(out) if out else ANALYSIS_DIR
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(cp.OUT / "mask_manifest.csv")
    if cases is not None:
        manifest = manifest[manifest.case_id.isin(set(cases))]
    n_cases, n_cells = len(manifest), int(manifest.masked_vars.str.count(",").add(1).sum())
    load = lambda path: (lambda t: t[t.case_id.isin(manifest.case_id)])(pd.read_csv(path).query("group_type=='all'"))
    paths = {REFERENCE_MODEL: ORIGINAL_ROOT / "evaluation" / REFERENCE_MODEL / "results.csv"}
    paths.update({m: eval_root / m / "results.csv" for m in MODELS})
    frames = {m: load(p) for m, p in paths.items()}
    keys = None
    for m, d in frames.items():
        assert len(d) == n_cells and d.case_id.nunique() == n_cases and not d.duplicated(["case_id", "variable"]).any(), m
        k = set(zip(d.case_id, d.variable))
        assert keys is None or k == keys, m
        keys = k
        assert (d.model == m).all(), m
    d = pd.concat(frames.values(), ignore_index=True)
    order = [REFERENCE_MODEL, *MODELS]
    sort = lambda t: t.assign(_o=t.model.map(order.index)).sort_values(["_o", *[c for c in t.columns if c in (
        "scenario", "gap_length_h", "variable")]]).drop(columns="_o")
    s = are.site_scores(d)
    s.pivot(index="greenhouse", columns="model", values="NMAE")[order].to_csv(out / "covreg_sites.csv")
    sort(are.summary(d)).to_csv(out / "covreg_summary.csv", index=False)
    sort(are.summary(d, extra=["scenario", "gap_length_h"])).to_csv(out / "covreg_scenario_gap.csv", index=False)
    sort(are.summary(d, extra=["scenario"])).to_csv(out / "covreg_scenario.csv", index=False)
    sort(are.summary(d, extra=["gap_length_h"])).to_csv(out / "covreg_gap.csv", index=False)
    sort(are.summary(d, "MAE", ["variable"])).to_csv(out / "covreg_variables_MAE.csv", index=False)
    sort(are.summary(d, "NMAE", ["variable"])).to_csv(out / "covreg_variables_NMAE.csv", index=False)
    pairs = [(REFERENCE_MODEL, m) for m in MODELS]
    are.compare(s, pairs).to_csv(out / "covreg_paired.csv", index=False)
    are.compare(s, [("CovReg-LGBM", "CovReg-Ridge")]).to_csv(out / "covreg_paired_lgbm_vs_ridge.csv", index=False)
    strata = []
    for extra in (["scenario"], ["gap_length_h"], ["scenario", "gap_length_h"], ["variable"]):
        for key, g in d.groupby(extra, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            c = are.compare(are.site_scores(g), pairs)
            for name, value in zip(extra, key):
                c[name] = value
            c["stratum"] = "x".join(extra)
            strata.append(c)
    pd.concat(strata, ignore_index=True).to_csv(out / "covreg_paired_strata.csv", index=False)
    # Refinement passes: CovReg passes 0,1,2,3,5 next to the production BiTFI passes.
    steps = []
    for m in MODELS:
        for k in SCORED_STEPS:
            q = load(eval_root / m / f"refine{k}.csv")
            assert len(q) == n_cells, (m, k)
            steps.append(q.assign(model=f"{m}-refine{k}", base=m, refinements=k))
    bitfi_steps = load(ORIGINAL_ROOT / "analysis/refinement_cases.csv")
    assert len(bitfi_steps) == n_cells * len(SCORED_STEPS)
    steps.append(bitfi_steps.assign(base=REFERENCE_MODEL))
    r = pd.concat(steps, ignore_index=True)
    meta = r[["model", "base", "refinements"]].drop_duplicates()
    are.summary(r).merge(meta, on="model").to_csv(out / "covreg_refinement.csv", index=False)
    are.summary(r, extra=["scenario"]).merge(meta, on="model").to_csv(out / "covreg_refinement_scenario.csv", index=False)
    step_pairs = []
    for prefix in (*MODELS, "BiTFI"):
        step_pairs += [(f"{prefix}-refine1", f"{prefix}-refine0"), (f"{prefix}-refine5", f"{prefix}-refine1")]
    are.compare(are.site_scores(r), step_pairs).to_csv(out / "covreg_refinement_paired.csv", index=False)
    by_scen = []
    for sc, g in r.groupby("scenario"):
        c = are.compare(are.site_scores(g), [p for p in step_pairs if p[0].endswith("refine5")])
        by_scen.append(c.assign(scenario=sc))
    pd.concat(by_scen, ignore_index=True).to_csv(out / "covreg_refinement_paired_scenario.csv", index=False)
    # Pass 0 is masked-input interpolation: it must reproduce the LI baseline cell by cell.
    li = load(ORIGINAL_ROOT / "evaluation/LI/results.csv")
    li_check = {}
    for m in MODELS:
        z = load(eval_root / m / "refine0.csv").merge(li, on=["case_id", "variable"], suffixes=("", "_LI"),
                                                     validate="one_to_one")
        li_check[m] = dict(cells=len(z), max_abs_NMAE_difference=float((z.NMAE - z.NMAE_LI).abs().max()),
                           max_abs_MAE_difference=float((z.MAE - z.MAE_LI).abs().max()))
    # Fallbacks and job structure.
    fb, diag = [], {}
    for m in MODELS:
        j = pd.read_csv(eval_root / m / "job_diagnostics.csv")
        j = j[j.case_id.isin(manifest.case_id)]
        for row in j.itertuples():
            for step, reason in enumerate(row.reasons.split(","), start=1):
                fb.append(dict(model=m, scenario=row.scenario, step=step, reason=reason))
        diag[m] = json.loads((eval_root / m / "complete.json").read_text())["diagnostics"]
    fbd = pd.DataFrame(fb).groupby(["model", "step", "reason", "scenario"]).size().rename("job_passes").reset_index()
    fbd.to_csv(out / "covreg_fallback.csv", index=False)
    # Context: the published representative configurations (original root, read only).
    ctx = [load(ORIGINAL_ROOT / "evaluation" / m / "results.csv").assign(model=m) for m in CONTEXT_MODELS]
    cdat = pd.concat([d, *ctx], ignore_index=True)
    are.summary(cdat).sort_values("mean").to_csv(out / "covreg_context_summary.csv", index=False)
    ctx_pairs = [(m, c) for m in MODELS for c in ("TimesFM3.0-COV-SPA", "SAITS-spatial", "AG-LightGBM", "Spatial-Ridge")]
    are.compare(are.site_scores(cdat), ctx_pairs).to_csv(out / "covreg_paired_context.csv", index=False)
    sources = {m: dict(path=str(p), sha256=sha256(p)) for m, p in paths.items()}
    copy_path = cp.OUT / "evaluation" / REFERENCE_MODEL / "results.csv"
    sources["BiTFI_copy_in_run_root"] = dict(path=str(copy_path), sha256=sha256(copy_path))
    sources["BiTFI_refinement_cases"] = dict(path=str(ORIGINAL_ROOT / "analysis/refinement_cases.csv"),
                                             sha256=sha256(ORIGINAL_ROOT / "analysis/refinement_cases.csv"))
    (out / "covreg_sources.json").write_text(json.dumps(dict(
        sources=sources, cases=n_cases, cells=n_cells, pass0_vs_LI=li_check, diagnostics=diag,
        context_models={m: str(ORIGINAL_ROOT / "evaluation" / m / "results.csv") for m in CONTEXT_MODELS},
        code_sha256=sha256(Path(__file__).resolve()),
        notes="site-level (greenhouse) n_eval-weighted means; bootstrap 10,000 site resamples (seed 42); "
              "reduction = 100*(1 - model/reference) on site means; Wilcoxon signed-rank on site scores; "
              "Holm within each compare() family; stratified p-values are descriptive"), indent=2, default=str))
    print(pd.read_csv(out / "covreg_summary.csv").to_string(), flush=True)
    print(pd.read_csv(out / "covreg_paired.csv").to_string(), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--overwrite", action="store_true", help="recompute this model's own evaluation folder")
    p.add_argument("--check-block", action="store_true", help="covariate-block identity check (.venv_tfm3, CPU)")
    p.add_argument("--analyze", action="store_true", help="first summary into analysis/revision2/covreg_*.csv")
    a = p.parse_args()
    np.random.seed(42)
    if a.check_block:
        run_block_check()
    elif a.analyze:
        run_analysis()
    elif a.model is None:
        p.error("--model is required for --smoke and evaluation runs")
    elif a.smoke:
        run_smoke(a.model)
    else:
        run_full(a.model, a.workers, a.overwrite)


if __name__ == "__main__":
    main()

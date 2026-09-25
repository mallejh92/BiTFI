"""
F6_imputation_examples.py
Figure 6 — 정성적 보간 예시 (시계열), Scenario A/B/C 동시.

레이아웃: 3행(Scenario A/B/C) × 5열(변수) = 15 패널 (legacy Fig3_timeseries 스타일).
  ─ 회색 실선  : observed context (갭 전/후)
  ─ 검정 점선  : ground truth (masked region)
  ─ 컬러 실선+마커 : 모델별 imputed (CAFI 강조)
  ─ 연파랑 음영 : gap 구간
  미마스킹 변수(예: Scenario B의 Tout/Rad)는 관측 covariate로 회색 실선만 표시.

공통 gap 구간(통일):
  시나리오마다 다른 시점을 쓰면 "환경"이 달라져 비교가 어렵다. 그래서 전 변수가
  관측된 **하나의 공통 gap 구간**(Scenario A 기준; 모든 변수에 유효)을 한 번 잡아
  A/B/C 모든 행·모든 변수에 동일하게 적용한다 → 행 간 차이는 오직 "마스킹 패턴".

서브플롯 주석:
  각 패널(마스킹된 변수)에 그 구간에서 가장 우수한 모델(최소 MAE)의 R²·MAE를 표기.

대표 온실:
  기본은 results.csv에서 전체 평균 NMAE에 가장 가까운 test 온실 자동 선택(--greenhouse로 지정).

입력 : 01_data/*.xlsx, split.json, results.csv, 03_result/comparison/models/
출력 : 03_result/comparison/figures/F6_imputation_{greenhouse}_ABC_gap{N}h.{pdf,png}

실행:
  python F6_imputation_examples.py --gap-length 72
  python F6_imputation_examples.py --greenhouse PF_0024693_01 --gap-length 72
"""

from __future__ import annotations

import json
import string
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
for _p in (_HERE, _MODEL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gpu_utils  # noqa: F401
import fig_utils as fu
from preprocessing import preprocess_file, TARGET_COLS
from masking_v2 import create_gap_masks, SCENARIO_CONFIGS, RANDOM_SEED
from evaluate import _inverse_transform

from models.linear_interpolation import LinearInterpolation
from models.seasonal_naive import SeasonalNaiveImputation
from models.foundation_model import ChronosImputation, TimesFMImputation, CAFIImputation

TARGET_VARS = list(TARGET_COLS.values())
AG_SPECS = {
    "AG-LightGBM":     {"RecursiveTabular": {"model_name": "GBM"}},
    "AG-RandomForest": {"RecursiveTabular": {"model_name": "RF"}},
    "AG-DeepAR":       {"DeepAR": {}},
    "AG-PatchTST":     {"PatchTST": {}},
}
DEFAULT_MODELS = ["LI", "SeasonalNaive", "AG-LightGBM", "Chronos2", "TimesFM2.5", "CAFI"]
SCENARIOS = ["A", "B", "C"]


def _stable_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) % 9999


# 모델별 최적 context(본문과 동일): zero-shot FM은 각자의 sweep 최적값을 쓴다.
# 여기 없는 모델(학습형 baseline 등)은 호출 시 넘어온 기본 context_len을 사용한다.
_MODEL_CONTEXT = {"Chronos2": 1080, "TimesFM2.5": 1080, "CAFI": 1440}


def _build_models(model_keys, context_len, models_dir):
    models = {}
    for k in model_keys:
        cl = _MODEL_CONTEXT.get(k, context_len)   # 모델별 context
        try:
            if k == "LI":
                models[k] = LinearInterpolation()
            elif k == "SeasonalNaive":
                models[k] = SeasonalNaiveImputation(season_length=24, context_len=cl)
            elif k in AG_SPECS:
                from models.autogluon_model import AutoGluonImputation
                mdir = models_dir / k
                if mdir.exists():
                    models[k] = AutoGluonImputation.load(
                        mdir, context_len=cl, hyperparameters=AG_SPECS[k], name=k)
                else:
                    warnings.warn(f"[{k}] 체크포인트 없음 → 생략")
            elif k == "Chronos2":
                models[k] = ChronosImputation(context_len=cl, fine_tune=False)
            elif k == "TimesFM2.5":
                models[k] = TimesFMImputation(context_len=cl)
            elif k == "CAFI":
                models[k] = CAFIImputation(fine_tune=False, context_len=cl, name="CAFI")
        except Exception as e:
            warnings.warn(f"[{k}] 생성 실패 → 생략: {e}")
    return models


def _pick_representative(rd: Path, test_stems: list[str]) -> str:
    try:
        df = fu.load_results(rd)
        a = df[df["group_type"] == "all"]
        per_gh = a.groupby("greenhouse").apply(fu.wavg_nmae).dropna()
        per_gh = per_gh[per_gh.index.isin(test_stems)]
        if per_gh.empty:
            raise ValueError
        gmean = per_gh.mean()
        rep = (per_gh - gmean).abs().idxmin()
        print(f"[F6] 대표 온실 = {rep} (NMAE={per_gh[rep]:.3f}, 전체평균={gmean:.3f})")
        return str(rep)
    except Exception as e:
        warnings.warn(f"대표 온실 선택 실패({e}) → 첫 test 온실 사용")
        return test_stems[0]


def _impute_at(models, full, masked_vars, gs, ge, tcols, ccols, scalers):
    """공통 윈도(gs,ge)에 masked_vars를 인공 결측 처리하고 모델별 보간(역정규화)."""
    masked = full.copy()
    eff = pd.DataFrame(np.where(full.isna().values, 0.0, 1.0).astype(np.float32),
                       index=full.index, columns=full.columns)
    for v in masked_vars:
        ci = full.columns.get_loc(v)
        masked.iloc[gs:ge + 1, ci] = np.nan
        eff.iloc[gs:ge + 1, ci] = 0.0
    preds = {}
    for k, m in models.items():
        # 메인 CAFI는 외부 환경 covariate(WindSpeed/RadIn)를 쓰지 않는다.
        # 변수 간 cross-variable refinement + 캘린더 covariate만 사용하므로
        # 모든 모델이 동일하게 5개 타깃 변수만 입력으로 받는다(공정 비교).
        masked_in = masked[tcols]
        mask_in = eff[tcols]
        try:
            out = m.impute(masked_in, mask_in)
            preds[k] = _inverse_transform(out[tcols], scalers)
        except Exception as e:
            warnings.warn(f"[{k}] impute 실패: {e}")
    return preds


def _metrics(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """gap 구간 R², MAE."""
    mae = float(np.mean(np.abs(true - pred)))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    return r2, mae


def _draw_panel(ax, var, gt_inv, preds, gs, ge, masked, model_names, context_display, full_len):
    gt = gt_inv[var].values
    gap_len = ge - gs + 1
    ctx_start_rel = -min(context_display, gs)
    post_end_rel = gap_len + min(24, full_len - ge - 1)
    x_ctx = np.arange(ctx_start_rel, 0); abs_ctx = gs + x_ctx
    x_gap = np.arange(0, gap_len)
    x_post = np.arange(gap_len, post_end_rel); abs_post = gs + x_post

    if not masked:
        x_all = np.arange(ctx_start_rel, post_end_rel); abs_all = gs + x_all
        ax.plot(x_all, gt[abs_all], color="#aaaaaa", linewidth=1.0, alpha=0.9)
        ax.text(0.5, 0.06, "observed covariate", transform=ax.transAxes, ha="center",
                fontsize=7, color="#888", style="italic")
        ax.set_xlim(ctx_start_rel, post_end_rel); ax.tick_params(labelsize=8)
        return

    ax.plot(x_ctx, gt[abs_ctx], color="#aaaaaa", linewidth=1.0, alpha=0.9, zorder=2)
    if len(x_post):
        ax.plot(x_post, gt[abs_post], color="#aaaaaa", linewidth=1.0, alpha=0.9, zorder=2)
    ax.plot(x_gap, gt[gs:ge + 1], color="#333333", linewidth=2.0, linestyle="--",
            alpha=0.85, zorder=6)

    true_gap = gt[gs:ge + 1]
    best = None  # (mae, r2, model)
    for k in model_names:
        if k not in preds or var not in preds[k].columns:
            continue
        pred_gap = preds[k][var].values[gs:ge + 1]
        ax.plot(x_gap, pred_gap, color=fu.cmp_model_color(k), linewidth=fu.cmp_lw(k),
                marker=fu.cmp_model_marker(k), markersize=3,
                markevery=max(1, gap_len // 12), zorder=7 if k == "CAFI" else 5)
        r2, mae = _metrics(true_gap, pred_gap)
        if best is None or mae < best[0]:
            best = (mae, r2, k)

    ax.axvspan(0, gap_len, color="#cce8ff", alpha=0.30, zorder=1)
    ax.axvline(0, color="#4499cc", linewidth=0.8, linestyle=":", zorder=3)
    ax.axvline(gap_len, color="#4499cc", linewidth=0.8, linestyle=":", zorder=3)
    ax.set_xlim(ctx_start_rel, post_end_rel); ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    # Best 모델(최소 MAE)의 R²·MAE 표기
    if best is not None:
        mae, r2, k = best
        r2_str = f"{r2:.2f}" if np.isfinite(r2) else "n/a"
        ax.text(0.97, 0.96,
                f"Best: {fu.cmp_model_display(k)}\n$R^2$={r2_str}, MAE={mae:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75,
                          edgecolor="#cccccc", linewidth=0.5), zorder=10)


def generate(result_dir=None, greenhouse=None, gap_length=72, repeat=0,
             context_display=168, context_len=720, model_keys=None) -> None:
    rd = Path(result_dir) if result_dir else fu.get_comparison_dir()
    split = json.load(open(rd / "split.json"))
    test_paths = [Path(p) for p in split["test"]]
    test_stems = [p.stem for p in test_paths]
    model_keys = model_keys or DEFAULT_MODELS

    if greenhouse is None:
        greenhouse = _pick_representative(rd, test_stems)
    if greenhouse not in test_stems:
        warnings.warn(f"{greenhouse}는 test 셋에 없음 → 첫 test 온실 사용. 가능: {test_stems}")
        greenhouse = test_stems[0]
    target_path = test_paths[test_stems.index(greenhouse)]

    res = preprocess_file(target_path, include_covariates=True)
    data = res["data"]; scalers = res["scaler"]; name = res["name"]
    tcols = [c for c in TARGET_VARS if c in data.columns]
    ccols = [c for c in data.columns if c not in TARGET_VARS]
    full = data[tcols + ccols]
    fvalid = pd.DataFrame(np.where(full.isna().values, 0.0, 1.0).astype(np.float32),
                          index=full.index, columns=full.columns)
    gt_inv = _inverse_transform(full[tcols], scalers)

    # ── 공통 gap 구간: 전 변수가 관측된 하나의 윈도(Scenario A 기준) ──
    common = create_gap_masks(full, fvalid, "C", gap_length, tcols, context_len,
                              n_repeats=repeat + 1,
                              random_seed=RANDOM_SEED + _stable_seed(f"{name}common{gap_length}"))
    if not common or repeat >= len(common):
        raise RuntimeError(f"공통 gap 구간 생성 실패: {name} gap{gap_length} "
                           f"(전 변수 관측 구간이 부족). 다른 온실/짧은 gap을 사용하세요.")
    gs, ge = common[repeat].gap_start_idx, common[repeat].gap_end_idx
    print(f"[F6] 공통 gap 구간: [{gs}, {ge}] (len={ge-gs+1})")

    models = _build_models(model_keys, context_len, rd / "models")
    model_names = fu.ordered_models(models.keys())

    # ── 시나리오별 보간 (동일 윈도, 마스킹 패턴만 변경) ──
    panel_data = {}  # (scenario, var) -> (preds, masked_bool)
    # Scenario A: 변수별 단일 마스킹
    for v in tcols:
        preds = _impute_at(models, full, [v], gs, ge, tcols, ccols, scalers)
        panel_data[("A", v)] = (preds, True)
    # Scenario B, C: 조합 마스킹 1회
    for scn in ("B", "C"):
        mvars = SCENARIO_CONFIGS[scn][0]
        preds = _impute_at(models, full, mvars, gs, ge, tcols, ccols, scalers)
        for v in tcols:
            panel_data[(scn, v)] = (preds, v in mvars)

    # ── 3×5 그리드 ──
    variables = [v for v in fu.VARIABLES if v in tcols]
    letters = list(string.ascii_lowercase)
    fig, axes = plt.subplots(len(SCENARIOS), len(variables),
                             figsize=(5.2 * len(variables), 3.4 * len(SCENARIOS)),
                             squeeze=False)
    idx = 0
    for ri, scn in enumerate(SCENARIOS):
        for ci, v in enumerate(variables):
            ax = axes[ri][ci]
            preds, masked = panel_data.get((scn, v), ({}, False))
            _draw_panel(ax, v, gt_inv, preds, gs, ge, masked, model_names,
                        context_display, len(full))
            if ri == 0:
                ax.set_title(fu.VARIABLE_LABELS.get(v, v), fontsize=10, pad=4)
            if ci == 0:
                ax.set_ylabel(f"{fu.SCENARIO_LABELS.get(scn, scn)}\n{fu.VARIABLE_UNITS.get(v, '')}",
                              fontsize=9)
            else:
                ax.set_ylabel(fu.VARIABLE_UNITS.get(v, ""), fontsize=8)
            if ri == len(SCENARIOS) - 1:
                ax.set_xlabel("Hours from gap start", fontsize=9)
            ax.text(0.03, 0.96, f"({letters[idx]})", transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="top")
            idx += 1

    legend_elems = [
        Line2D([0], [0], color="#aaaaaa", linewidth=1.5, label="Observed context"),
        Line2D([0], [0], color="#333333", linewidth=2.0, linestyle="--",
               label="Ground truth (masked region)"),
        Patch(facecolor="#cce8ff", alpha=0.5, label="Gap region"),
    ] + [
        Line2D([0], [0], color=fu.cmp_model_color(k), marker=fu.cmp_model_marker(k),
               markersize=6, linewidth=1.6, label=fu.cmp_model_display(k))
        for k in model_names
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=min(len(legend_elems), 6),
               fontsize=9, bbox_to_anchor=(0.5, -0.03), frameon=True,
               framealpha=0.9, edgecolor="#cccccc")
    fig.suptitle(f"Imputation examples — {name}  |  Scenarios A/B/C (common gap window)  "
                 f"|  Gap = {gap_length} h", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fu.save_figure(fig, rd / "figures", f"F6_imputation_{name}_ABC_gap{gap_length}h")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="F6 보간 예시 시계열 (Scenario A/B/C, 공통 윈도)")
    p.add_argument("--result-dir", type=str, default=str(fu.get_comparison_dir()))
    p.add_argument("--greenhouse", type=str, default=None,
                   help="미지정 시 results.csv 전체 평균 NMAE에 가장 가까운 test 온실 자동 선택")
    p.add_argument("--gap-length", type=int, default=72)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--context-display", type=int, default=168)
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    args = p.parse_args()
    mk = [m.strip() for m in args.models.split(",") if m.strip()]
    generate(Path(args.result_dir), greenhouse=args.greenhouse, gap_length=args.gap_length,
             repeat=args.repeat, context_display=args.context_display, model_keys=mk)

"""
Fig. 3 — 시계열 보간 예시

특정 온실·갭에 대해 각 모델의 imputation 결과를 시각화한다.

Layout: 1행 × 5열(변수) = 1×5 서브플롯 (가로 배열)
  ─ 회색 실선         : context (갭 이전 관측값)
  ─ 회색 점선(두꺼움)  : ground truth (갭 구간 실측값)
  ─ 컬러 실선         : 각 모델의 imputed values
  ─ 연한 파란 배경     : 갭 구간

실행:
  python fig3_timeseries.py
  python fig3_timeseries.py --greenhouse PF_0020209_01 --scenario C --gap-length 72
  python fig3_timeseries.py --context-display 168 --skip-patchtst
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_LEGACY_DIR = _HERE.parent
_MODEL_DIR = _LEGACY_DIR.parent
_FIGURES_DIR = _MODEL_DIR / "figures"
for _p in (_HERE, _MODEL_DIR, _FIGURES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gpu_utils  # noqa: F401
import fig_utils as fu

from preprocessing import preprocess_file, train_test_split, fit_scalers_from_split
from masking_v2 import create_gap_masks, SCENARIO_CONFIGS
from evaluate import _inverse_transform


# ─── 모델 빌더 ──────────────────────────────────────────────────────────────
def _build_and_fit(
    keys: list[str],
    train_norm: pd.DataFrame,
    context_len: int = 720,
) -> dict:
    from models.linear_interpolation import LinearInterpolation
    from models.seasonal_naive import SeasonalNaiveImputation
    from models.autogluon_model import AutoGluonImputation
    from models.foundation_model import ChronosImputation, CAFIImputation

    models: dict = {}
    for k in keys:
        k = k.lower()
        t0 = time.perf_counter()
        try:
            if k == "linear":
                m = LinearInterpolation(); m.fit(train_norm); models["LinearInterp"] = m
            elif k == "seasonal":
                m = SeasonalNaiveImputation(season_length=24, context_len=context_len)
                m.fit(train_norm); models["SeasonalNaive"] = m
            elif k == "recursivetabular":
                m = AutoGluonImputation(
                    context_len=context_len, time_limit=120,
                    hyperparameters={"RecursiveTabular": {}}, name="RecursiveTabular",
                )
                m.fit(train_norm); models["RecursiveTabular"] = m
            elif k == "patchtst":
                from models.patchtst_model import PatchTSTImputation
                m = PatchTSTImputation(
                    context_window=context_len, patch_len=24,
                    d_model=128, d_ff=256, mask_type="mixed",
                    epochs=50, patience=10,
                )
                m.fit(train_norm); models["PatchTST"] = m
            elif k == "chronos2":
                m = ChronosImputation(context_len=context_len, fine_tune=False)
                m.fit(train_norm); models["Chronos2"] = m
            elif k == "cafi":
                m = CAFIImputation(context_len=context_len, max_rounds=5)
                m.fit(train_norm); models["CAFI"] = m
        except Exception as e:
            warnings.warn(f"[{k}] 실패: {e}")
        print(f"  [{k}] 준비 완료 ({time.perf_counter()-t0:.1f}s)")
    return models


# ─── 그리기 ─────────────────────────────────────────────────────────────────
def draw(
    mask_result,
    imputed_dict: dict[str, pd.DataFrame],
    scalers: dict,
    gap_start: int,
    gap_end: int,
    context_display: int,
    out_dir: Path,
    scenario: str,
    gap_length_h: int,
    greenhouse: str,
) -> None:
    fu.setup_figure_style()

    model_names = list(imputed_dict.keys())
    variables   = fu.VARIABLES          # ['Tin','Tout','RH','CO2','Rad']

    # 1행 × 5열: 변수 5개를 가로 방향으로 배열
    fig, axes = plt.subplots(
        1, 5,
        figsize=(28, 5),
        squeeze=False,
    )  # axes.shape == (1, 5)

    # ── 역정규화 ──
    gt_phys = _inverse_transform(mask_result.ground_truth, scalers)
    imputed_phys: dict[str, pd.DataFrame] = {
        nm: _inverse_transform(df, scalers) for nm, df in imputed_dict.items()
    }

    # x 범위 (갭 시작 = 0 기준)
    ctx_start_rel = -min(context_display, gap_start)
    post_end_rel  = gap_length_h + min(24, len(mask_result.ground_truth) - gap_end - 1)

    x_ctx  = np.arange(ctx_start_rel, 0)
    x_gap  = np.arange(0, gap_length_h)
    x_post = np.arange(gap_length_h, post_end_rel)
    abs_ctx  = gap_start + x_ctx
    abs_post = gap_start + x_post

    import string
    alpha_labels = list(string.ascii_lowercase)

    for idx, var in enumerate(variables):
        ax = axes[0, idx]
        gt_arr = gt_phys[var].values if var in gt_phys.columns else None

        if gt_arr is not None:
            # ─ context (갭 이전 관측값, 회색 실선)
            ax.plot(x_ctx, gt_arr[abs_ctx],
                    color="#aaaaaa", linewidth=1.0, alpha=0.9, zorder=2)
            # ─ after-gap context
            if len(x_post):
                ax.plot(x_post, gt_arr[abs_post],
                        color="#aaaaaa", linewidth=1.0, alpha=0.9, zorder=2)
            # ─ ground truth in gap (회색 점선, 두꺼움)
            ax.plot(x_gap, gt_arr[gap_start:gap_end + 1],
                    color="#333333", linewidth=2.0, linestyle="--",
                    alpha=0.85, zorder=6)

        # ─ 각 모델 imputed (컬러 실선)
        for model_name in model_names:
            if model_name not in imputed_phys:
                continue
            if var not in imputed_phys[model_name].columns:
                continue
            color  = fu.MODEL_COLORS.get(model_name, "#444444")
            marker = fu.MODEL_MARKERS.get(model_name, "o")
            imp_gap = imputed_phys[model_name][var].values[gap_start:gap_end + 1]
            ax.plot(x_gap, imp_gap,
                    color=color, linewidth=1.6, zorder=5,
                    marker=marker, markersize=3, markevery=max(1, gap_length_h // 12))

        # ─ 갭 배경 & 경계선
        ax.axvspan(0, gap_length_h, color="#cce8ff", alpha=0.30, zorder=1)
        ax.axvline(0,            color="#4499cc", linewidth=0.8, linestyle=":", zorder=3)
        ax.axvline(gap_length_h, color="#4499cc", linewidth=0.8, linestyle=":", zorder=3)

        # ─ 축 꾸미기
        unit = fu.VARIABLE_UNITS.get(var, "")
        ax.set_title(fu.VARIABLE_LABELS[var], fontsize=10)
        ax.set_ylabel(unit, fontsize=9)
        ax.set_xlabel("Hours from gap start", fontsize=9)
        ax.set_xlim(ctx_start_rel, post_end_rel)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

        ax.text(0.02, 0.97, f"({alpha_labels[idx]})",
                transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top")

    # ── 하단 공통 범례 ──
    legend_elems = [
        Line2D([0],[0], color="#aaaaaa", linewidth=1.5,
               label="Observed context"),
        Line2D([0],[0], color="#333333", linewidth=2.0, linestyle="--",
               label="Ground truth (masked region)"),
        Patch(facecolor="#cce8ff", alpha=0.5, label="Gap region"),
    ] + [
        Line2D(
            [0],[0],
            color=fu.MODEL_COLORS.get(m, "#444444"),
            marker=fu.MODEL_MARKERS.get(m, "o"),
            markersize=6, linewidth=1.6,
            label=fu.MODEL_DISPLAY.get(m, m),
        )
        for m in model_names
    ]
    fig.legend(
        handles=legend_elems,
        loc="lower center",
        ncol=len(legend_elems),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    scenario_label = fu.SCENARIO_LABELS.get(scenario, scenario).replace("\n", " ")
    fig.suptitle(
        f"Imputation example — {greenhouse}  |  {scenario_label}  |  Gap = {gap_length_h} h",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    fname = f"Fig3_timeseries_{greenhouse}_Sc{scenario}_gap{gap_length_h}h"
    fu.save_figure(fig, out_dir, fname)
    plt.close(fig)


# ─── 메인 ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Fig 3 — time series imputation example")
    parser.add_argument("--greenhouse", default="PF_0020209_01")
    parser.add_argument("--scenario",   default="C",
                        choices=["A", "B", "C"])
    parser.add_argument("--gap-length", type=int, default=72)
    parser.add_argument("--repeat",     type=int, default=0,
                        help="몇 번째 반복 갭을 시각화할지")
    parser.add_argument("--context-display", type=int, default=168,
                        help="갭 이전 표시할 context 길이 (h)")
    parser.add_argument("--context-len", type=int, default=720)
    parser.add_argument("--skip-patchtst", action="store_true", default=False)
    parser.add_argument("--data-dir",   default=None)
    parser.add_argument("--out-dir",    default=None)
    args = parser.parse_args()

    fu.setup_figure_style()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else fu.get_data_dir()
    out_dir  = Path(args.out_dir).resolve()  if args.out_dir  else fu.get_figure_dir()

    # ── 데이터 로드 ──
    fp = data_dir / f"{args.greenhouse}.xlsx"
    if not fp.exists():
        fps = sorted(data_dir.glob("PF_*.xlsx"))
        fp  = fps[0] if fps else None
    if fp is None or not fp.exists():
        print("데이터 파일 없음"); return

    print(f"[fig3] 데이터: {fp.name}")
    result = preprocess_file(fp)
    if result is None: return

    train, test = train_test_split(result, test_ratio=0.2)
    scalers, train_norm, test_norm = fit_scalers_from_split(
        train["data_raw"], test["data_raw"]
    )
    original_valid_mask = pd.DataFrame(
        np.where(test_norm.isna().values, 0.0, 1.0).astype(np.float32),
        index=test_norm.index, columns=test_norm.columns,
    )

    # ── 갭 생성 ──
    var_combos = SCENARIO_CONFIGS[args.scenario]
    masked_vars = var_combos[0]  # scenario C → all vars; A → Tin; B → indoor
    masks = create_gap_masks(
        test_data=test_norm,
        original_valid_mask=original_valid_mask,
        scenario=args.scenario,
        gap_length_h=args.gap_length,
        masked_vars=masked_vars,
        context_len_h=args.context_len,
        n_repeats=args.repeat + 1,
        random_seed=42,
    )
    if not masks or args.repeat >= len(masks):
        print(f"repeat={args.repeat} 불가 (생성된 갭={len(masks)}개)"); return

    mr = masks[args.repeat]
    print(f"[fig3] 갭 인덱스: [{mr.gap_start_idx}, {mr.gap_end_idx}]  "
          f"masked_vars={mr.masked_vars}")

    # ── 모델 학습 ──
    model_keys = ["linear", "seasonal", "recursivetabular", "chronos2", "cafi"]
    if not args.skip_patchtst:
        model_keys.insert(3, "patchtst")

    print(f"[fig3] 모델 학습 중 ({len(model_keys)}개)...")
    models = _build_and_fit(model_keys, train_norm, args.context_len)

    # ── 보간 ──
    imputed_dict: dict[str, pd.DataFrame] = {}
    for name, model in models.items():
        print(f"[fig3] 보간: {name} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            imp = model.impute(mr.masked_data, mr.effective_mask)
            imputed_dict[name] = imp
            print(f"{time.perf_counter()-t0:.1f}s")
        except Exception as e:
            print(f"실패: {e}")

    # ── 그리기 ──
    draw(
        mask_result=mr,
        imputed_dict=imputed_dict,
        scalers=scalers,
        gap_start=mr.gap_start_idx,
        gap_end=mr.gap_end_idx,
        context_display=args.context_display,
        out_dir=out_dir,
        scenario=args.scenario,
        gap_length_h=args.gap_length,
        greenhouse=args.greenhouse,
    )


if __name__ == "__main__":
    main()

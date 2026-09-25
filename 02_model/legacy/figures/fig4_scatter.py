"""
Fig. 4 — Measured vs Predicted 산점도

각 모델에 대해 여러 갭의 실측값(Measured) vs 예측값(Predicted)을 산점도로 시각화한다.

Layout: 2행 × 3열 = 6 패널 (모델별)
  ─ x축: Measured (original scale)
  ─ y축: Predicted (original scale)
  ─ 점 색상: 변수별
  ─ 검은 대각선: 완벽 예측 기준선 (y = x)
  ─ 패널 내 표시: R², RMSE

데이터: 하나의 온실, 모든 시나리오·갭·반복 조합

실행:
  python fig4_scatter.py
  python fig4_scatter.py --greenhouse PF_0020209_01 --scenario A
  python fig4_scatter.py --skip-patchtst --gap-lengths 24,72,168
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
from masking_v2 import create_gap_masks, SCENARIO_CONFIGS, GAP_LENGTHS_H
from evaluate import _inverse_transform


# ─── 모델 빌더 (fig3_timeseries와 동일) ─────────────────────────────────────
def _build_and_fit(keys: list[str], train_norm: pd.DataFrame, context_len: int = 720) -> dict:
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


# ─── 예측값 수집 ─────────────────────────────────────────────────────────────
def collect_predictions(
    models: dict,
    test_norm: pd.DataFrame,
    original_valid_mask: pd.DataFrame,
    scalers: dict,
    scenarios: list[str],
    gap_lengths_h: list[int],
    n_repeats: int,
    context_len: int,
) -> pd.DataFrame:
    """
    모든 (시나리오 × 갭 × 반복 × 모델 × 변수)에 대해
    eval 위치의 (measured, predicted) 쌍을 수집한다.
    """
    rows = []

    for scenario in scenarios:
        var_combos = SCENARIO_CONFIGS[scenario]
        for masked_vars in var_combos:
            for gap_h in gap_lengths_h:
                masks = create_gap_masks(
                    test_data=test_norm,
                    original_valid_mask=original_valid_mask,
                    scenario=scenario,
                    gap_length_h=gap_h,
                    masked_vars=masked_vars,
                    context_len_h=context_len,
                    n_repeats=n_repeats,
                    random_seed=42 + hash(f"{scenario}{gap_h}") % 999,
                )
                if not masks:
                    continue

                for repeat_idx, mr in enumerate(masks):
                    print(
                        f"  [scatter] sc={scenario} gap={gap_h}h "
                        f"vars={masked_vars} repeat={repeat_idx}", end=" "
                    )
                    # 역정규화된 ground truth
                    gt_phys = _inverse_transform(mr.ground_truth, scalers)

                    for model_name, model in models.items():
                        try:
                            imp_norm = model.impute(mr.masked_data, mr.effective_mask)
                        except Exception as e:
                            continue
                        imp_phys = _inverse_transform(imp_norm, scalers)

                        # eval 위치: eval_mask == True
                        for var in fu.VARIABLES:
                            if var not in mr.eval_mask.columns:
                                continue
                            emask = mr.eval_mask[var].values.astype(bool)
                            if emask.sum() == 0:
                                continue

                            measured  = gt_phys[var].values[emask]
                            predicted = imp_phys[var].values[emask]

                            valid = (
                                ~np.isnan(measured) & ~np.isnan(predicted) &
                                ~np.isinf(measured) & ~np.isinf(predicted)
                            )
                            measured  = measured[valid]
                            predicted = predicted[valid]
                            if len(measured) == 0:
                                continue

                            for m_val, p_val in zip(measured, predicted):
                                rows.append({
                                    "model":     model_name,
                                    "scenario":  scenario,
                                    "gap_h":     gap_h,
                                    "variable":  var,
                                    "repeat":    repeat_idx,
                                    "measured":  m_val,
                                    "predicted": p_val,
                                })
                    print("✓")

    return pd.DataFrame(rows)


# ─── 그리기 ─────────────────────────────────────────────────────────────────
TARGET_MODELS = ["SeasonalNaive", "CAFI"]


def draw(
    pred_df: pd.DataFrame,
    out_dir: Path,
    greenhouse: str,
) -> None:
    fu.setup_figure_style()

    variables = fu.VARIABLES   # ['Tin','Tout','RH','CO2','Rad']
    models    = [m for m in TARGET_MODELS if m in pred_df["model"].unique()]

    # 5행(변수) × 2열(모델) 그리드
    # 레이블 순서 (행 우선): a=SN Tin, b=CAFI Tin, c=SN Tout, d=CAFI Tout, …
    n_row, n_col = len(variables), len(models)
    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(n_col * 5.5, n_row * 5),
        squeeze=False,
    )

    import string
    panel_labels = list(string.ascii_lowercase)  # a–j

    label_idx = 0
    for row_idx, var in enumerate(variables):
        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx][col_idx]

            sub = pred_df[
                (pred_df["model"] == model_name) &
                (pred_df["variable"] == var)
            ]
            if sub.empty:
                ax.axis("off")
                label_idx += 1
                continue

            color = fu.VARIABLE_COLORS.get(var, "#444444")
            ax.scatter(
                sub["measured"], sub["predicted"],
                c=color,
                s=8, alpha=0.4, linewidths=0,
                rasterized=True,
            )

            # y=x 기준선
            combined = pd.concat([sub["measured"], sub["predicted"]])
            vmin, vmax = combined.min(), combined.max()
            margin = (vmax - vmin) * 0.05
            lim = (vmin - margin, vmax + margin)
            ax.plot(lim, lim, color="#333333", linewidth=1.2, linestyle="--", zorder=10)
            ax.set_xlim(lim); ax.set_ylim(lim)

            # R² / RMSE
            y_true = sub["measured"].values
            y_pred = sub["predicted"].values
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            ax.text(
                0.05, 0.93,
                f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}",
                transform=ax.transAxes, fontsize=9,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="none"),
            )

            # 열 상단에만 모델명 표시
            if row_idx == 0:
                ax.set_title(fu.MODEL_DISPLAY.get(model_name, model_name),
                             fontsize=11, fontweight="bold")

            # 좌측 열에만 변수 라벨 표시
            if col_idx == 0:
                ax.set_ylabel(
                    f"{fu.VARIABLE_LABELS[var]}\nPredicted",
                    fontsize=9,
                )
            else:
                ax.set_ylabel("Predicted", fontsize=9)

            # 하단 행에만 x축 라벨 표시
            if row_idx == n_row - 1:
                ax.set_xlabel("Measured", fontsize=9)

            ax.tick_params(labelsize=8)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

            ax.text(0.02, 0.98, f"({panel_labels[label_idx]})",
                    transform=ax.transAxes, fontsize=9,
                    fontweight="bold", va="top")
            label_idx += 1

    fig.suptitle(
        f"Measured vs Predicted — {greenhouse}  (all scenarios / gap lengths / greenhouses)",
        fontsize=11, y=1.005,
    )
    fig.tight_layout()

    fu.save_figure(fig, out_dir, "Fig4_scatter_all")
    plt.close(fig)


# ─── 갭 길이 고정 산점도 (모든 모델) ────────────────────────────────────────
def draw_by_gap(pred_df: pd.DataFrame, gap_h: int, out_dir: Path) -> None:
    """특정 갭 길이의 (measured, predicted) 쌍을 모든 모델에 대해 산점도로 그린다.

    Layout: 2행 × 3열 (모델별 패널, 변수는 색상으로 구분)
    """
    fu.setup_figure_style()

    sub_df = pred_df[pred_df["gap_h"] == gap_h]
    if sub_df.empty:
        print(f"[fig4] gap_h={gap_h} 데이터 없음 — skip")
        return

    model_order = [m for m in fu.MODEL_ORDER if m in sub_df["model"].unique()]
    n_col = 3
    n_row = int(np.ceil(len(model_order) / n_col))

    import string
    labels = list(string.ascii_lowercase)

    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(n_col * 5.5, n_row * 5),
        squeeze=False,
    )

    for idx, model_name in enumerate(model_order):
        ri, ci = divmod(idx, n_col)
        ax = axes[ri][ci]

        msub = sub_df[sub_df["model"] == model_name]
        if msub.empty:
            ax.axis("off"); continue

        for var in fu.VARIABLES:
            vsub = msub[msub["variable"] == var]
            if vsub.empty:
                continue
            ax.scatter(
                vsub["measured"], vsub["predicted"],
                c=fu.VARIABLE_COLORS[var],
                s=8, alpha=0.4, linewidths=0,
                label=fu.VARIABLE_LABELS[var].split("(")[0].strip(),
                rasterized=True,
            )

        combined = pd.concat([msub["measured"], msub["predicted"]])
        vmin, vmax = combined.min(), combined.max()
        margin = (vmax - vmin) * 0.05
        lim = (vmin - margin, vmax + margin)
        ax.plot(lim, lim, color="#333333", linewidth=1.2, linestyle="--", zorder=10)
        ax.set_xlim(lim); ax.set_ylim(lim)

        y_true = msub["measured"].values
        y_pred = msub["predicted"].values
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        ax.text(
            0.05, 0.93,
            f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f}",
            transform=ax.transAxes, fontsize=9,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="none"),
        )

        ax.set_title(fu.MODEL_DISPLAY.get(model_name, model_name),
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Measured", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        ax.text(0.02, 0.98, f"({labels[idx]})",
                transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    # 사용하지 않는 서브플롯 숨김
    for idx in range(len(model_order), n_row * n_col):
        ri, ci = divmod(idx, n_col)
        axes[ri][ci].axis("off")

    # 공통 범례 (변수 색상)
    from matplotlib.lines import Line2D as _L
    var_handles = [
        _L([0], [0], marker="o", color="w",
           markerfacecolor=fu.VARIABLE_COLORS[v], markersize=7,
           label=fu.VARIABLE_LABELS[v].split("(")[0].strip())
        for v in fu.VARIABLES
    ] + [_L([0], [0], color="#333333", linewidth=1.2, linestyle="--", label="y = x")]
    fig.legend(
        handles=var_handles,
        loc="lower center",
        ncol=len(var_handles),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )

    fig.suptitle(
        f"Measured vs Predicted — Gap = {gap_h} h  (all scenarios / greenhouses)",
        fontsize=11, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.99))

    fu.save_figure(fig, out_dir, f"Fig4_scatter_gap{gap_h}h")
    plt.close(fig)


# ─── 메인 ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Fig 4 — Measured vs Predicted scatter (all greenhouses)")
    parser.add_argument("--scenarios",    default="A,B,C")
    parser.add_argument("--gap-lengths",  default="6,24,72,168",
                        help="쉼표 구분 gap length (h)")
    parser.add_argument("--n-repeats",    type=int, default=5,
                        help="갭 반복 수 (scatter 점 수 결정)")
    parser.add_argument("--context-len",  type=int, default=720)
    parser.add_argument("--skip-patchtst", action="store_true", default=False)
    parser.add_argument("--gap-filter", default="6",
                        help="draw_by_gap 으로 따로 그릴 갭 길이 (쉼표 구분, 예: '6' 또는 '6,24'). 빈 문자열이면 skip.")
    parser.add_argument("--data-dir",  default=None)
    parser.add_argument("--out-dir",   default=None)
    args = parser.parse_args()

    fu.setup_figure_style()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else fu.get_data_dir()
    out_dir  = Path(args.out_dir).resolve()  if args.out_dir  else fu.get_figure_dir()

    scenarios   = [s.strip() for s in args.scenarios.split(",")]
    gap_lengths = [int(g) for g in args.gap_lengths.split(",")]
    model_keys  = ["linear", "seasonal", "recursivetabular", "chronos2", "cafi"]
    if not args.skip_patchtst:
        model_keys.insert(3, "patchtst")

    # ── 온실별 루프 ──
    all_dfs: list[pd.DataFrame] = []
    for fp in sorted(data_dir.glob("PF_*.xlsx")):
        print(f"\n[fig4] ── {fp.name} ──")
        result = preprocess_file(fp)
        if result is None:
            continue

        train, test = train_test_split(result, test_ratio=0.2)
        scalers, train_norm, test_norm = fit_scalers_from_split(
            train["data_raw"], test["data_raw"]
        )
        original_valid_mask = pd.DataFrame(
            np.where(test_norm.isna().values, 0.0, 1.0).astype(np.float32),
            index=test_norm.index, columns=test_norm.columns,
        )

        print(f"[fig4] 모델 학습 중 ({len(model_keys)}개)...")
        models = _build_and_fit(model_keys, train_norm, args.context_len)

        print("[fig4] 예측값 수집 중...")
        gh_df = collect_predictions(
            models=models,
            test_norm=test_norm,
            original_valid_mask=original_valid_mask,
            scalers=scalers,
            scenarios=scenarios,
            gap_lengths_h=gap_lengths,
            n_repeats=args.n_repeats,
            context_len=args.context_len,
        )
        if not gh_df.empty:
            gh_df["greenhouse"] = fp.stem
            all_dfs.append(gh_df)
            print(f"[fig4] {fp.stem}: {len(gh_df)}쌍 수집")

    if not all_dfs:
        print("[fig4] 수집된 예측값 없음"); return

    pred_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n[fig4] 전체 {len(pred_df)}쌍 수집 완료 ({pred_df['greenhouse'].nunique()}개 온실)")

    # ── 전체 산점도 ──
    draw(pred_df, out_dir, "All greenhouses")

    # ── 갭 길이별 산점도 (모든 모델) ──
    if args.gap_filter.strip():
        for gh in [g.strip() for g in args.gap_filter.split(",") if g.strip()]:
            draw_by_gap(pred_df, int(gh), out_dir)


if __name__ == "__main__":
    main()

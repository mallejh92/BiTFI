"""
Fig. 2 — 보간 성능 비교 (v2, experiment_v2 결과 기반)

Layout: 3행(Scenario A/B/C) × 5열(변수) = 15 서브플롯
  x축: Gap length (6, 12, 24, 72, 168 h)
  y축: RMSE (원래 단위, 온실 × 반복 평균 ± SD)
  선:  6개 모델 (LinearInterp, SeasonalNaive, RecursiveTabular,
                 PatchTST, Chronos2, CAFI)

실행:
  python fig2_imputation_results.py
  python fig2_imputation_results.py --result-dir ../../03_result/experiment_v2
                                    --out-dir ../../04_figure
  python fig2_imputation_results.py --metric nRMSE   # nRMSE 버전
"""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
_LEGACY_DIR = _HERE.parent
_MODEL_DIR = _LEGACY_DIR.parent
_FIGURES_DIR = _MODEL_DIR / "figures"
for _p in (_HERE, _MODEL_DIR, _FIGURES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fig_utils as fu


# ── 상수 ─────────────────────────────────────────────────────────────────────
SCENARIOS  = ["A", "B", "C"]
VARIABLES  = fu.VARIABLES          # ['Tin','Tout','RH','CO2','Rad']
GAP_LENGTHS = fu.GAP_LENGTHS_H     # [6, 12, 24, 72, 168]


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_results(result_dir: Path) -> pd.DataFrame | None:
    path = result_dir / "results.csv"
    if not path.exists():
        # 구버전 경로 fallback
        path = result_dir / "results_all.csv"
    if not path.exists():
        print(f"[fig2] 결과 파일 없음: {result_dir}")
        return None
    df = pd.read_csv(path)
    print(f"[fig2] {len(df)}행 로드: {path}")
    return df


# ── 집계 ─────────────────────────────────────────────────────────────────────
def aggregate(
    df: pd.DataFrame,
    scenario: str,
    variable: str,
    metric: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    (scenario, variable) 조건에서 모델별 gap_length_h → (mean, std) 반환.
    온실 × 반복을 모집단으로 사용한다.

    Returns
    -------
    dict: model → (mean_array, std_array)  각각 len(GAP_LENGTHS)
    """
    sub = df[(df["scenario"] == scenario) & (df["variable"] == variable)]
    if sub.empty:
        return {}

    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model, g in sub.groupby("model"):
        means, stds = [], []
        for gap in GAP_LENGTHS:
            vals = g[g["gap_length_h"] == gap][metric].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std()  if len(vals) > 1 else 0.0)
        result[str(model)] = (np.array(means), np.array(stds))
    return result


def _model_sort_key(name: str) -> int:
    try:
        return fu.MODEL_ORDER.index(name)
    except ValueError:
        return len(fu.MODEL_ORDER)


# ── 그리기 ────────────────────────────────────────────────────────────────────
def draw(df: pd.DataFrame, out_dir: Path, metric: str = "RMSE") -> None:
    n_row = len(SCENARIOS)
    n_col = len(VARIABLES)

    # 폭을 넓게 — 모델 6개, 변수 5개, 시나리오 3개
    fig, axes = plt.subplots(
        n_row, n_col,
        figsize=(20, 11),
        squeeze=False,
        sharex=True,
    )

    # 데이터에 등장하는 모델 정렬
    present_models = sorted(df["model"].astype(str).unique(), key=_model_sort_key)

    x_pos   = np.arange(len(GAP_LENGTHS))
    x_ticks = [fu.GAP_LABELS[g] for g in GAP_LENGTHS]
    labels  = list(string.ascii_lowercase)
    subplot_idx = 0

    for ri, scenario in enumerate(SCENARIOS):
        for ci, var in enumerate(VARIABLES):
            ax = axes[ri][ci]
            agg = aggregate(df, scenario, var, metric)

            for model in present_models:
                if model not in agg:
                    continue
                means, stds = agg[model]
                color  = fu.MODEL_COLORS.get(model, "#444444")
                marker = fu.MODEL_MARKERS.get(model, "o")
                label  = fu.MODEL_DISPLAY.get(model, model)

                ax.plot(
                    x_pos, means,
                    color=color, marker=marker, markersize=5,
                    linewidth=1.6, label=label, zorder=3,
                )
                std_safe = np.where(np.isnan(stds), 0.0, stds)
                ax.fill_between(
                    x_pos, means - std_safe, means + std_safe,
                    color=color, alpha=0.12, linewidth=0, zorder=1,
                )

            # ── 축 꾸미기 ──
            if ri == 0:
                ax.set_title(fu.VARIABLE_LABELS[var], fontsize=10, pad=4)

            if ri == n_row - 1:
                ax.set_xlabel("Gap length", fontsize=10)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_ticks, fontsize=8)
            else:
                ax.set_xticks(x_pos)
                ax.set_xticklabels([])

            if ci == 0:
                scenario_label = fu.SCENARIO_LABELS.get(scenario, scenario)
                ax.set_ylabel(
                    f"{scenario_label}\n{metric}",
                    fontsize=9, labelpad=4,
                )
            else:
                ax.set_ylabel("")

            ax.tick_params(labelsize=8)
            ax.set_xlim(-0.4, len(GAP_LENGTHS) - 0.6)

            # 데이터 없는 서브플롯 (해당 시나리오에서 결측 대상이 아닌 변수)
            if not agg:
                ax.set_facecolor("#f5f5f5")
                ax.text(
                    0.5, 0.5,
                    "Observable covariate\n(not masked\nin this scenario)",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    fontsize=8, color="#666666", style="italic",
                    multialignment="center",
                )
                ax.tick_params(left=False, labelleft=False)
            else:
                ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
                # y 최솟값 0으로 고정
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(max(0, ymin * 0.95), ymax * 1.05)

            # 서브플롯 라벨 (a)–(o)
            ax.text(
                0.03, 0.96, f"({labels[subplot_idx]})",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="left",
            )
            subplot_idx += 1

    # ── 공통 범례 (하단 중앙) ──
    legend_handles = [
        Line2D(
            [0], [0],
            color=fu.MODEL_COLORS.get(m, "#444444"),
            marker=fu.MODEL_MARKERS.get(m, "o"),
            markersize=6, linewidth=1.8,
            label=fu.MODEL_DISPLAY.get(m, m),
        )
        for m in present_models
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )

    unit = fu.VARIABLE_UNITS.get("Tin", "original units")
    fig.suptitle(
        f"Imputation performance by gap length  ({metric}, mean ± SD across greenhouses × repetitions)",
        fontsize=12, y=1.005,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.99))

    fname = f"Fig2_imputation_results_{metric}"
    fu.save_figure(fig, out_dir, fname)
    plt.close(fig)

    # ── 보조: nRMSE 버전도 자동 생성 (RMSE 요청 시) ──
    if metric == "RMSE" and "nRMSE" in df.columns:
        draw(df, out_dir, metric="nRMSE")


def draw_placeholder(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.axis("off")
    ax.text(
        0.5, 0.5,
        "Results not yet available\n(results.csv not found)",
        ha="center", va="center", fontsize=22, color="#b2182b",
    )
    fu.save_figure(fig, out_dir, "Fig2_imputation_results")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Fig 2 — imputation results (v2)")
    parser.add_argument(
        "--result-dir", default=None,
        help="결과 디렉토리 (기본: <root>/03_result/experiment_v2)",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="그림 출력 디렉토리 (기본: <root>/04_figure)",
    )
    parser.add_argument(
        "--metric", default="RMSE", choices=["RMSE", "nRMSE", "MAE", "R2"],
        help="y축 지표 (기본: RMSE)",
    )
    args = parser.parse_args()

    fu.setup_figure_style()

    result_dir = (
        Path(args.result_dir).resolve() if args.result_dir
        else fu.get_result_dir("experiment_v2")
    )
    out_dir = (
        Path(args.out_dir).resolve() if args.out_dir
        else fu.get_figure_dir()
    )

    print(f"[fig2] result_dir = {result_dir}")
    print(f"[fig2] out_dir    = {out_dir}")
    print(f"[fig2] metric     = {args.metric}")

    df = load_results(result_dir)
    if df is None or df.empty:
        draw_placeholder(out_dir)
        return

    draw(df, out_dir, metric=args.metric)


if __name__ == "__main__":
    main()

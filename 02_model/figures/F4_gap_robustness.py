"""
F4_gap_robustness.py
Figure 4 — 갭 길이에 따른 보간 성능 (mean ± SD).

레이아웃·스타일은 legacy `fig2_imputation_results.py`(Fig2_imputation_results_nRMSE)와
동일하다: 3행(Scenario A/B/C) × 5열(변수), x=gap length(범주형), 모델별 평균선 +
온실×반복에 대한 ±SD 밴드, 미마스킹 변수는 covariate placeholder.

지표: 기본은 MAE. F4는 변수별 패널(5열)이라 각 칸이 단일 변수·단일 단위이므로
      정규화가 불필요하고, 오히려 NMAE는 gap이 커질수록 평가구간의 값 범위(분모)가
      넓어져 오차가 커져도 NMAE가 작아지는 아티팩트를 만든다(gap-dependent bias).
      단위 없는 비교가 필요하면 --metric RMSE(=√MSE) 사용. NMAE는 변수 통합 비교
      (F3/F5)에서만 의미가 있으며 여기선 권장하지 않는다.

입력 : 03_result/comparison/results.csv  (group_type == 'all')
출력 : 03_result/comparison/figures/F4_gap_robustness_{metric}.{pdf,png}

실행:
  python F4_gap_robustness.py                 # MAE (기본)
  python F4_gap_robustness.py --metric RMSE
"""

from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import fig_utils as fu

SCENARIOS = ["A", "B", "C"]
VARIABLES = fu.VARIABLES
GAP_LENGTHS = fu.GAP_LENGTHS_H


def _prepare_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """results.csv에 없는 지표는 가능한 경우 파생한다(RMSE=√MSE)."""
    df = df.copy()
    if metric not in df.columns:
        if metric == "RMSE" and "MSE" in df.columns:
            df["RMSE"] = np.sqrt(df["MSE"].clip(lower=0))
        else:
            raise ValueError(f"지표 '{metric}'를 results.csv에서 찾거나 파생할 수 없습니다. "
                             f"사용 가능: NMAE, MAE, RMSE(=√MSE)")
    return df


def aggregate(df, scenario, variable, metric):
    """(scenario, variable)에서 모델별 gap → (mean, std). 온실×반복을 모집단으로."""
    sub = df[(df["scenario"] == scenario) & (df["variable"] == variable)]
    if sub.empty:
        return {}
    out = {}
    for model, g in sub.groupby("model"):
        means, stds = [], []
        for gap in GAP_LENGTHS:
            vals = g[g["gap_length_h"] == gap][metric].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std() if len(vals) > 1 else 0.0)
        out[str(model)] = (np.array(means), np.array(stds))
    return out


def draw(df, out_dir, metric="NMAE"):
    n_row, n_col = len(SCENARIOS), len(VARIABLES)
    fig, axes = plt.subplots(n_row, n_col, figsize=(20, 11), squeeze=False, sharex=True)
    present = fu.ordered_models(df["model"].astype(str).unique())
    x_pos = np.arange(len(GAP_LENGTHS))
    x_ticks = [fu.GAP_LABELS[g] for g in GAP_LENGTHS]
    letters = list(string.ascii_lowercase)
    idx = 0

    for ri, scenario in enumerate(SCENARIOS):
        for ci, var in enumerate(VARIABLES):
            ax = axes[ri][ci]
            agg = aggregate(df, scenario, var, metric)
            for model in present:
                if model not in agg:
                    continue
                means, stds = agg[model]
                color = fu.cmp_model_color(model)
                ax.plot(x_pos, means, color=color, marker=fu.cmp_model_marker(model),
                        markersize=5, linewidth=fu.cmp_lw(model),
                        label=fu.cmp_model_display(model),
                        zorder=4 if model == "CAFI" else 3)
                std_safe = np.where(np.isnan(stds), 0.0, stds)
                ax.fill_between(x_pos, means - std_safe, means + std_safe,
                                color=color, alpha=0.12, linewidth=0, zorder=1)

            if ri == 0:
                ax.set_title(fu.VARIABLE_LABELS[var], fontsize=10, pad=4)
            if ri == n_row - 1:
                ax.set_xlabel("Gap length", fontsize=10)
                ax.set_xticks(x_pos); ax.set_xticklabels(x_ticks, fontsize=8)
            else:
                ax.set_xticks(x_pos); ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel(f"{fu.SCENARIO_LABELS.get(scenario, scenario)}\n{metric}",
                              fontsize=9, labelpad=4)
            ax.tick_params(labelsize=8)
            ax.set_xlim(-0.4, len(GAP_LENGTHS) - 0.6)

            if not agg:  # 해당 시나리오에서 마스킹되지 않는 변수
                ax.set_facecolor("#f5f5f5")
                ax.text(0.5, 0.5, "Observable covariate\n(not masked\nin this scenario)",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="#666666", style="italic", multialignment="center")
                ax.tick_params(left=False, labelleft=False)
            else:
                ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(max(0, ymin * 0.95), ymax * 1.05)

            ax.text(0.03, 0.96, f"({letters[idx]})", transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="top", ha="left")
            idx += 1

    handles = [Line2D([0], [0], color=fu.cmp_model_color(m), marker=fu.cmp_model_marker(m),
                      markersize=6, linewidth=1.8, label=fu.cmp_model_display(m))
               for m in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=10, bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle(f"Imputation performance by gap length  "
                 f"({metric}, mean ± SD across greenhouses × repetitions)",
                 fontsize=12, y=1.005)
    fig.tight_layout(rect=(0, 0.05, 1, 0.99))
    fu.save_figure(fig, out_dir, f"F4_gap_robustness_{metric}")
    plt.close(fig)


def generate(result_dir=None, metric="MAE"):
    df = fu.load_results(result_dir)
    df = df[df["group_type"] == "all"].copy()
    if df.empty:
        raise ValueError("group_type='all' 행이 없습니다.")
    df = _prepare_metric(df, metric)
    out_dir = (Path(result_dir) if result_dir else fu.get_comparison_dir()) / "figures"
    draw(df, out_dir, metric=metric)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="F4 갭 길이 강건성 (fig2 형식)")
    p.add_argument("--result-dir", type=str, default=str(fu.get_comparison_dir()))
    p.add_argument("--metric", type=str, default="MAE", choices=["MAE", "RMSE", "NMAE"])
    args = p.parse_args()
    generate(Path(args.result_dir), metric=args.metric)

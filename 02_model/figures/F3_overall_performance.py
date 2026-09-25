"""
F3_overall_performance.py
Figure 3 — 전체 성능 비교 (main result).

입력 : 03_result/comparison/results.csv  (group_type == 'all')
출력 : 03_result/comparison/figures/F3_overall_performance.{pdf,png}

레이아웃:
  (a) 모델별 전체 NMAE 막대 (온실 간 평균 ± SD), 오름차순, CAFI 강조
  (b) 모델 × 변수 NMAE 히트맵 (n_eval 가중)

실행:
  python F3_overall_performance.py
  python F3_overall_performance.py --result-dir ../../03_result/comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import fig_utils as fu


def _overall_panel(ax, allrows: pd.DataFrame, models: list[str]) -> None:
    # 모델별: 온실 단위 NMAE를 먼저 구하고, 온실 간 평균±SD로 불확실성을 표시한다.
    means, sds = [], []
    for m in models:
        sub = allrows[allrows["model"] == m]
        per_gh = sub.groupby("greenhouse", observed=True).apply(fu.wavg_nmae).dropna()
        means.append(per_gh.mean() if len(per_gh) else np.nan)
        sds.append(per_gh.std() if len(per_gh) > 1 else 0.0)
    order = np.argsort(means)
    models_s = [models[i] for i in order]
    means_s = [means[i] for i in order]
    sds_s = [sds[i] for i in order]
    colors = [fu.cmp_model_color(m) for m in models_s]
    y = np.arange(len(models_s))
    ax.barh(y, means_s, xerr=sds_s, color=colors, edgecolor="black", linewidth=0.6,
            error_kw=dict(elinewidth=0.8, capsize=2))
    ax.set_yticks(y)
    ax.set_yticklabels([fu.cmp_model_display(m) for m in models_s])
    ax.invert_yaxis()  # 가장 좋은(작은) 모델이 위로
    ax.set_xlabel("NMAE")
    ax.set_title("(a) Overall NMAE by model")
    # 라벨이 오차막대 끝(mean+SD) 오른쪽에 오도록 x 한계에 여백을 둔다.
    bar_ends = [(mu if np.isnan(sd) else mu + sd) for mu, sd in zip(means_s, sds_s)]
    xmax = max([b for b in bar_ends if not np.isnan(b)] or [1.0])
    pad = 0.015 * xmax
    ax.set_xlim(0, xmax * 1.18)
    for yi, (mu, sd, m) in enumerate(zip(means_s, sds_s, models_s)):
        end = mu if np.isnan(sd) else mu + sd
        ax.text(end + pad, yi, f"{mu:.3f}", va="center", ha="left", fontsize=8,
                fontweight="bold" if m == "CAFI" else "normal")


def _variable_heatmap(ax, allrows: pd.DataFrame, models: list[str]) -> None:
    vars = [v for v in fu.VARIABLES if v in allrows["variable"].unique()]
    mat = np.full((len(models), len(vars)), np.nan)
    for i, m in enumerate(models):
        for j, v in enumerate(vars):
            mat[i, j] = fu.wavg_nmae(allrows[(allrows["model"] == m) & (allrows["variable"] == v)])
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd_r", vmin=np.nanmin(mat), vmax=np.nanpercentile(mat, 95))
    ax.set_xticks(range(len(vars)))
    ax.set_xticklabels([fu.VAR_SHORT_LABELS.get(v, v) for v in vars])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([fu.cmp_model_display(m) for m in models])
    ax.set_title("(b) NMAE by model × variable")
    for i in range(len(models)):
        for j in range(len(vars)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=12,
                        color="white" if mat[i, j] < np.nanpercentile(mat, 70) else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="NMAE")


def generate(result_dir: Path | None = None) -> None:
    df = fu.load_results(result_dir)
    allrows = df[df["group_type"] == "all"].copy()
    if allrows.empty:
        raise ValueError("group_type='all' 행이 없습니다.")
    models = fu.ordered_models(allrows["model"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(models) + 3.5),
                             gridspec_kw={"width_ratios": [1.1, 1.0]})
    _overall_panel(axes[0], allrows, models)
    _variable_heatmap(axes[1], allrows, models)
    fig.tight_layout()
    out_dir = (Path(result_dir) if result_dir else fu.get_comparison_dir()) / "figures"
    fu.save_figure(fig, out_dir, "F3_overall_performance")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="F3 전체 성능 비교")
    p.add_argument("--result-dir", type=str, default=str(fu.get_comparison_dir()))
    args = p.parse_args()
    generate(Path(args.result_dir))

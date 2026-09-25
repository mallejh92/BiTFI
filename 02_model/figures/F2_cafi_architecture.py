"""
F2_cafi_architecture.py
Figure 2 — CAFI 방법 모식도 (개념도, 데이터 불필요).

CAFI = Covariate-Aware Foundation Imputation
  Round 0 : 파운데이션 backbone(Chronos-2)으로 변수별 단변량 초기 보간
  Round k : 결측 변수 X를 나머지 변수(관측/이전 추정)를 past/future covariate로
            주어 backbone의 cross-variate attention으로 재예측 → 수렴까지 반복

출력 : 03_result/comparison/figures/F2_cafi_architecture.{pdf,png}
실행 : python F2_cafi_architecture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import fig_utils as fu


def _box(ax, xy, w, h, text, fc, fontsize=10, ec="black"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3, wrap=True)


def _arrow(ax, p0, p1, style="-|>", color="black", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                                 lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def generate(result_dir: Path | None = None) -> None:
    # 논문에서 전폭(figure*, \textwidth)으로 삽입되므로, 좁은 \columnwidth를
    # 가정했던 이전 크기(11x5.5)보다 작은 native 캔버스를 쓰되 폰트를 키워
    # 최종 인쇄 크기가 다른 그림들과 맞도록 한다.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    col_in, col_r0, col_rk, col_out = "#EEEEEE", "#D9E8F5", "#F5E1EC", "#DFF0DF"
    fs = 11

    # 입력
    _box(ax, (0.3, 2.6), 2.2, 1.8,
         "Masked multivariate\nseries\n($T_{in},T_{out}$,RH,CO$_2$,Rad)\n+ calendar\n+ gap mask", col_in, fs)
    # Round 0
    _box(ax, (3.2, 4.2), 2.6, 1.4,
         "Round 0\nFoundation backbone\n(per-variable, univariate)", col_r0, fs)
    _box(ax, (3.2, 1.0), 2.6, 1.4,
         "Foundation model\n(Chronos-2)\ncross-variate attention", "#FFFFFF", fs, ec="#666666")
    # Round k
    _box(ax, (6.5, 2.6), 3.0, 1.8,
         "Round k (k≥1)\nfor each gap variable X:\n  covariates = other vars\n  X ← backbone(X | covariates)", col_rk, fs)
    # 수렴/출력
    _box(ax, (10.0, 2.9), 1.7, 1.2, "Converged?\nΔ < tol", "#FFF6D9", fs)
    _box(ax, (9.7, 0.4), 2.2, 1.1, "Imputed series\n(evaluated on\nartificial gaps)", col_out, fs)

    # 화살표
    _arrow(ax, (2.5, 3.8), (3.2, 4.7))                       # 입력 → Round0
    _arrow(ax, (4.5, 4.2), (4.5, 2.4), color="#666666")      # Round0 → backbone
    _arrow(ax, (5.8, 3.5), (6.5, 3.5))                       # → Round k
    _arrow(ax, (4.5, 1.0), (6.6, 2.6), color="#666666", rad=-0.2)  # backbone → Round k
    _arrow(ax, (9.5, 3.5), (10.0, 3.5))                      # Round k → 수렴
    _arrow(ax, (10.85, 4.1), (8.0, 4.4), color="#888888", rad=0.35)  # no → 반복
    ax.text(9.2, 4.6, "no → next round", fontsize=fs - 1, color="#888888")
    _arrow(ax, (10.85, 2.9), (10.8, 1.5))                    # yes → 출력
    ax.text(11.0, 2.2, "yes", fontsize=fs - 1)

    ax.set_title("CAFI: Covariate-Aware Foundation Imputation", fontsize=fs + 4, pad=8)
    fig.tight_layout()
    out_dir = (Path(result_dir) if result_dir else fu.get_comparison_dir()) / "figures"
    fu.save_figure(fig, out_dir, "F2_cafi_architecture")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="F2 CAFI 아키텍처 모식도")
    p.add_argument("--result-dir", type=str, default=str(fu.get_comparison_dir()))
    args = p.parse_args()
    generate(Path(args.result_dir))

"""
F8_ablation.py
Figure 8 — CAFI 구성요소 ablation.

입력 : 03_result/comparison/ablation.csv   (variant = CAFI-full/noExtCov/r1/r2/r3)
        03_result/comparison/results.csv    (Chronos2 = refinement 0회 = backbone)
출력 : 03_result/comparison/figures/F8_ablation.{pdf,png}

레이아웃 (1×2):
  (a) Cross-variable refinement 수렴 곡선
       x = refinement round 수(0=Chronos-2 backbone, 1·2·3, 5=CAFI-full),
       y = 전체 NMAE. 1회에서 대부분의 이득이 발생하고 2~3회에서 수렴함을 보인다.
  (b) 구성요소 분해(변수별 grouped bar)
       Chronos-2(backbone) → +cross-variable refinement(noExtCov)
       → +external covariates(CAFI-full). 외부 covariate(WindSpeed/RadIn)의
       기여가 어떤 변수에서 큰지(일사·습도) 드러난다.

집계는 다른 피규어와 동일하게 fu.wavg_nmae(n_eval 가중)를 사용한다.

실행:
  python F8_ablation.py
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

# refinement round 수 → ablation variant 이름(round 0은 backbone = 메인 비교의 Chronos2).
# 메인 CAFI는 외부 covariate를 쓰지 않으므로 수렴 곡선도 ext_cov=False(noExt) 변형을 쓴다.
_ROUND_VARIANT = {1: "CAFI-noExt-r1", 2: "CAFI-noExt-r2", 3: "CAFI-noExt-r3", 5: "CAFI-noExtCov"}
_CAFI_FINAL = "CAFI-noExtCov"          # 외부 covariate 없는 최종 CAFI
_BACKBONE = "Chronos2"                 # results.csv 상의 이름(= refinement 0회)
_VAR_LABELS = {"Tin": r"$T_{in}$", "Tout": r"$T_{out}$", "RH": "RH",
               "CO2": r"CO$_2$", "Rad": "Rad"}


def _load_ablation(result_dir: Path | None = None) -> pd.DataFrame:
    rd = Path(result_dir) if result_dir else fu.get_comparison_dir()
    path = rd / "ablation.csv"
    if not path.exists():
        raise FileNotFoundError(f"ablation.csv 없음: {path}. 먼저 run_ablation.py 실행.")
    return pd.read_csv(path)


def _convergence_panel(ax, ab_all: pd.DataFrame, backbone_nmae: float) -> None:
    """(a) refinement round 수 vs 전체 NMAE 수렴 곡선."""
    rounds = [0, 1, 2, 3, 5]
    ys = [backbone_nmae]
    for r in rounds[1:]:
        ys.append(fu.wavg_nmae(ab_all[ab_all["model"] == _ROUND_VARIANT[r]]))
    ax.plot(rounds, ys, marker="o", markersize=7, linewidth=1.8,
            color=fu.cmp_model_color("CAFI"), markeredgecolor="black",
            markeredgewidth=0.5, zorder=3)
    # backbone(round 0) 지점 강조 + 라벨(제목과 겹치지 않게 오른쪽 아래로)
    ax.scatter([0], [ys[0]], s=90, color=fu.cmp_model_color(_BACKBONE),
               edgecolor="black", linewidth=0.6, zorder=4)
    ax.annotate("Chronos-2 backbone\n(0 rounds)", xy=(0.08, ys[0]),
                xytext=(1.35, ys[0] - 0.0010), fontsize=8, va="top", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.4"))
    # round 0→1 감소율 주석
    drop = 100 * (ys[0] - ys[1]) / ys[0]
    ax.annotate(f"$-${drop:.0f}%", xy=(0.55, (ys[0] + ys[1]) / 2),
                fontsize=9, fontweight="bold", color="black", ha="left")
    for r, y in zip(rounds, ys):
        va = "bottom" if r == 0 else "top"
        off = 0.0009 if r == 0 else -0.0009
        ax.text(r, y + off, f"{y:.3f}", ha="center", va=va, fontsize=7.5)
    ax.set_ylim(top=ys[0] + 0.0032)
    ax.set_xticks(rounds)
    ax.set_xlabel("Cross-variable refinement rounds")
    ax.set_ylabel("Overall NMAE (n$_{eval}$-weighted)")
    ax.set_title("(a) Refinement convergence")
    ax.margins(x=0.08)


def _decomposition_panel(ax, main_all: pd.DataFrame) -> None:
    """(b) 변수별: backbone(Chronos-2) → CAFI(cross-variable refinement).

    backbone·CAFI 모두 메인 results.csv에서 읽는다(메인 CAFI는 외부 covariate
    미사용 = cross-variable refinement 전용이므로 별도 ablation.csv가 불필요).
    """
    vars_ = [v for v in fu.VARIABLES if v in main_all["variable"].unique()]
    steps = [
        ("Chronos-2 (backbone)", fu.cmp_model_color(_BACKBONE), lambda v: fu.wavg_nmae(
            main_all[(main_all["model"] == _BACKBONE) & (main_all["variable"] == v)])),
        ("+ cross-variable refinement (CAFI)", fu.cmp_model_color("CAFI"), lambda v: fu.wavg_nmae(
            main_all[(main_all["model"] == "CAFI") & (main_all["variable"] == v)])),
    ]
    x = np.arange(len(vars_))
    width = 0.8 / len(steps)
    for i, (label, color, fn) in enumerate(steps):
        ys = [fn(v) for v in vars_]
        ax.bar(x + i * width, ys, width, label=label, color=color,
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x + width * (len(steps) - 1) / 2)
    ax.set_xticklabels([_VAR_LABELS.get(v, v) for v in vars_])
    ax.set_ylabel("NMAE")
    ax.set_title("(b) Refinement gain by variable")
    ax.legend(fontsize=7.5, loc="upper left")


def generate(result_dir: Path | None = None) -> None:
    main = fu.load_results(result_dir)
    main_all = main[main["group_type"] == "all"].copy()
    backbone_nmae = fu.wavg_nmae(main_all[main_all["model"] == _BACKBONE])
    if not np.isfinite(backbone_nmae):
        raise ValueError(f"backbone({_BACKBONE}) 결과가 results.csv에 없습니다.")

    # 외부 covariate를 제거한 뒤 CAFI의 유일한 요소는 cross-variable refinement이므로
    # ablation은 변수별 backbone→refinement 단일 패널로 제시한다.
    fig, ax = plt.subplots(figsize=(8, 5))
    _decomposition_panel(ax, main_all)
    ax.set_title("Ablation: cross-variable refinement gain by variable")
    fig.tight_layout()
    out_dir = (Path(result_dir) if result_dir else fu.get_comparison_dir()) / "figures"
    fu.save_figure(fig, out_dir, "F8_ablation")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="F8 CAFI ablation")
    p.add_argument("--result-dir", type=str, default=str(fu.get_comparison_dir()))
    args = p.parse_args()
    generate(Path(args.result_dir))

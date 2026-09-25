"""
FS1_context_sweep.py
Supplementary Figure S1 — context-length sensitivity of the three zero-shot
foundation models (Chronos-2, TimesFM-2.5, CAFI).

입력 : 03_result/context_sweep/ctx{C}/results.csv   (C ∈ CONTEXTS)
        각 파일은 run_comparison.py를 축소 프로토콜(Scenario A, gap 24h+168h,
        10 test 온실, 3 repeat)로 context_len=C에서 실행한 결과.
출력 : 03_result/comparison/figures/FS1_context_sweep.{pdf,png}

그림 : x = context length (h, 로그 눈금), y = overall NMAE(n_eval 가중),
       모델별 선. context_len=720(본문 설정)을 수직 점선으로 표시.
       각 모델 최소 NMAE 지점을 마커로 강조.

실행:
  python FS1_context_sweep.py
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

_CONTEXTS = [96, 168, 336, 720, 1080, 1440, 1900]
_MODELS = ["Chronos2", "TimesFM2.5", "CAFI"]
_MAIN_CONTEXT = 720


def _sweep_dir() -> Path:
    return fu.get_comparison_dir().parent / "context_sweep"


def _load_curve() -> pd.DataFrame:
    """(model, context) → overall NMAE 테이블."""
    rows = []
    sd = _sweep_dir()
    for c in _CONTEXTS:
        csv = sd / f"ctx{c}" / "results.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        allr = df[df["group_type"] == "all"]
        for m in _MODELS:
            v = fu.wavg_nmae(allr[allr["model"] == m])
            if np.isfinite(v):
                rows.append({"context": c, "model": m, "nmae": v})
    return pd.DataFrame(rows)


def generate(result_dir: Path | None = None) -> pd.DataFrame:
    curve = _load_curve()
    if curve.empty:
        raise FileNotFoundError(
            f"context_sweep 결과 없음: {_sweep_dir()}. 먼저 run_ctx_sweep.sh 실행."
        )

    fig, ax = plt.subplots(figsize=(7.2, 5))
    for m in _MODELS:
        sub = curve[curve["model"] == m].sort_values("context")
        if sub.empty:
            continue
        ax.plot(sub["context"], sub["nmae"], marker=fu.cmp_model_marker(m),
                markersize=7, linewidth=fu.cmp_lw(m), color=fu.cmp_model_color(m),
                markeredgecolor="black", markeredgewidth=0.4,
                label=fu.cmp_model_display(m))
        # 최소 지점 강조
        imin = sub["nmae"].values.argmin()
        ax.scatter([sub["context"].values[imin]], [sub["nmae"].values[imin]],
                   s=140, facecolors="none", edgecolors=fu.cmp_model_color(m),
                   linewidths=1.8, zorder=5)

    # 최적 대역(1080–1440h; FM 최적 1080, CAFI 최적 1440)을 음영으로 표시.
    ax.axvspan(1080, 1440, color="0.85", alpha=0.5, zorder=0)
    ax.text(np.sqrt(1080 * 1440), ax.get_ylim()[1], "optimal band\n(1080–1440 h)",
            va="top", ha="center", fontsize=8, color="0.4")

    ax.set_xscale("log")
    ax.set_xticks(_CONTEXTS)
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Context length (h)")
    ax.set_ylabel("Overall NMAE (n$_{eval}$-weighted)")
    ax.set_title("Context-length sensitivity (zero-shot models)\n"
                 "Scenario A, gaps 24 h + 168 h, 10 test greenhouses",
                 fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_dir = (Path(result_dir) if result_dir else fu.get_comparison_dir()) / "figures"
    fu.save_figure(fig, out_dir, "FS1_context_sweep")
    plt.close(fig)

    # 최적 context 요약 출력(720 최적 여부 판단용)
    print("\n=== context별 overall NMAE ===")
    piv = curve.pivot(index="context", columns="model", values="nmae")
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== 모델별 최적 context & 720 대비 ===")
    for m in _MODELS:
        sub = curve[curve["model"] == m].set_index("context")["nmae"]
        if sub.empty:
            continue
        best_c = sub.idxmin(); best_v = sub.min()
        v720 = sub.get(_MAIN_CONTEXT, np.nan)
        gain = 100 * (v720 - best_v) / v720 if np.isfinite(v720) and v720 > 0 else np.nan
        print(f"  {m:<12} best={best_c}h ({best_v:.4f})  720h={v720:.4f}  "
              f"720 대비 최적이 {gain:+.1f}% 낮음")
    return curve


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="FS1 context-length sensitivity")
    p.add_argument("--result-dir", type=str, default=str(fu.get_comparison_dir()))
    args = p.parse_args()
    generate(Path(args.result_dir))

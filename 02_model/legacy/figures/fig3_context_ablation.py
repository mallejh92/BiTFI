"""
Fig. 3 — Context Length 어블레이션
Context length가 CAFI와 AutoGluon 성능에 미치는 영향.

원예학적 해석 annotation:
  24h:  "Daily mgmt."
  168h: "Weekly cycle"
  672h: "Seasonal"

Layout (1-row, 2-column, 14×6 inches):
  Panel A: Mean MAE vs context_len for CAFI and AutoGluon (and TimesFM for reference)
           Each variable shown as separate thin line, overall mean as thick line
  Panel B: Inference time vs context_len for each model (log scale on y)

실행: python fig3_context_ablation.py [--result-dir ../../03_result/context_ablation]
                                       [--out-dir ../../04_figure]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_HERE = Path(__file__).resolve().parent
_LEGACY_DIR = _HERE.parent
_MODEL_DIR = _LEGACY_DIR.parent
_FIGURES_DIR = _MODEL_DIR / "figures"
for _p in (_HERE, _MODEL_DIR, _FIGURES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fig_utils as fu  # noqa: E402


# Panel A 에 표시할 모델 (존재하는 것만 자동 필터)
PANEL_A_MODELS = ["CAFI", "AutoGluon", "TimesFM"]

# 원예학적 해석 주석 (context_len → 라벨)
HORTI_ANNOTATIONS = {
    24:  "Daily mgmt.\n일일 관리",
    168: "Weekly cycle\n주간 관리",
    672: "Seasonal\n계절 전환",
}


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
def load_csv(result_dir: Path, name: str) -> pd.DataFrame | None:
    path = result_dir / name
    if not path.exists():
        print(f"[fig3] {name} 없음: {path}")
        return None
    df = pd.read_csv(path)
    print(f"[fig3] loaded {len(df)} rows from {path}")
    return df


def _present_context_lens(df: pd.DataFrame) -> list[int]:
    if df is None or "context_len" not in df.columns:
        return fu.CONTEXT_LENS
    present = sorted(int(c) for c in df["context_len"].dropna().unique())
    return present or fu.CONTEXT_LENS


# ---------------------------------------------------------------------------
# Panel A: MAE vs context_len
# ---------------------------------------------------------------------------
def draw_panel_a(ax, df: pd.DataFrame, ctx_lens: list[int]) -> None:
    ax.text(-0.02, 1.05, "(A)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom", ha="left")
    ax.set_title("Imputation accuracy vs context length", loc="left")

    models = [m for m in PANEL_A_MODELS
              if m in df["model"].astype(str).unique()]
    if not models:  # 지정 모델이 없으면 데이터에 존재하는 모델 사용
        models = sorted(df["model"].astype(str).unique())

    x = np.arange(len(ctx_lens))

    for model in models:
        msub = df[df["model"].astype(str) == model]
        mcolor = fu.MODEL_COLORS.get(model, "#444444")

        # 변수별 얇은 선
        for v in fu.VARIABLES:
            vsub = msub[msub["variable"] == v]
            if vsub.empty:
                continue
            means = (vsub.groupby("context_len")["MAE"].mean()
                     .reindex(ctx_lens))
            ax.plot(
                x, means.values, color=fu.VARIABLE_COLORS[v],
                linewidth=0.8, alpha=0.4, zorder=1,
            )

        # 전체 평균 굵은 선 (모델 색)
        overall = (msub.groupby("context_len")["MAE"].mean()
                   .reindex(ctx_lens))
        ax.plot(
            x, overall.values, color=mcolor, linewidth=2.6,
            marker=fu.MODEL_MARKERS.get(model, "o"), markersize=6,
            label=model, zorder=3,
        )

    # 원예학적 해석 수직 점선 + 주석
    ymax = ax.get_ylim()[1]
    for ctx, note in HORTI_ANNOTATIONS.items():
        if ctx in ctx_lens:
            xi = ctx_lens.index(ctx)
            ax.axvline(xi, color="#777777", linestyle="--",
                       linewidth=0.9, alpha=0.7, zorder=0)
            ax.text(
                xi, ymax * 0.98, note, rotation=0, fontsize=8,
                ha="center", va="top", color="#444444",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#cccccc", alpha=0.85),
            )

    ax.set_xticks(x)
    ax.set_xticklabels([fu.CONTEXT_LABELS.get(c, str(c)) for c in ctx_lens],
                       fontsize=8)
    ax.set_xlabel("Context Length", fontsize=11)
    ax.set_ylabel("Mean MAE", fontsize=12)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    # 범례: 모델 굵은 선 + 얇은 선 설명
    handles = [
        Line2D([0], [0], color=fu.MODEL_COLORS.get(m, "#444444"),
               linewidth=2.6, marker=fu.MODEL_MARKERS.get(m, "o"),
               markersize=6, label=m)
        for m in models
    ]
    handles.append(
        Line2D([0], [0], color="#999999", linewidth=0.8, alpha=0.6,
               label="thin lines = individual variables")
    )
    ax.legend(handles=handles, fontsize=8, loc="best")


# ---------------------------------------------------------------------------
# Panel B: inference time vs context_len
# ---------------------------------------------------------------------------
def draw_panel_b(ax, df: pd.DataFrame, ctx_lens: list[int]) -> None:
    ax.text(-0.02, 1.05, "(B)", transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom", ha="left")
    ax.set_title("Inference cost vs context length", loc="left")

    # 추론 시간 컬럼 선택 (impute_seconds 우선)
    time_col = None
    for c in ["impute_seconds", "total_seconds", "fit_seconds"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        ax.text(0.5, 0.5, "No timing column available",
                ha="center", va="center", transform=ax.transAxes,
                color="#b2182b")
        return

    x = np.arange(len(ctx_lens))
    models = sorted(df["model"].astype(str).unique(),
                    key=lambda m: (fu.MODEL_ORDER.index(m)
                                   if m in fu.MODEL_ORDER else 99))

    any_positive = False
    for model in models:
        msub = df[df["model"].astype(str) == model]
        color = fu.MODEL_COLORS.get(model, "#444444")
        grp = msub.groupby("context_len")[time_col].agg(["mean", "std"])
        grp = grp.reindex(ctx_lens)
        mean = grp["mean"].values
        std = np.where(np.isnan(grp["std"].values), 0.0, grp["std"].values)

        ax.plot(x, mean, color=color, linewidth=1.8,
                marker=fu.MODEL_MARKERS.get(model, "o"), markersize=5,
                label=model, zorder=3)
        lower = np.clip(mean - std, a_min=1e-6, a_max=None)
        ax.fill_between(x, lower, mean + std, color=color, alpha=0.15,
                        linewidth=0, zorder=1)
        if np.nanmax(mean) and np.nanmax(mean) > 0:
            any_positive = True

    ax.set_xticks(x)
    ax.set_xticklabels([fu.CONTEXT_LABELS.get(c, str(c)) for c in ctx_lens],
                       fontsize=8)
    ax.set_xlabel("Context Length", fontsize=11)
    ax.set_ylabel(f"Inference Time (s)  [{time_col}]", fontsize=12)
    if any_positive:
        ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.5,
            alpha=0.5)
    ax.legend(fontsize=8, loc="best")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def draw_placeholder(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    ax.text(0.5, 0.5,
            "Results not yet available\n(ablation CSV not found)",
            ha="center", va="center", fontsize=20, color="#b2182b")
    fu.save_figure(fig, out_dir, "Fig3_context_ablation")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fig 3 — context ablation")
    parser.add_argument("--result-dir", default=None,
                        help="결과 디렉토리 (기본: <root>/03_result/context_ablation)")
    parser.add_argument("--out-dir", default=None,
                        help="그림 출력 디렉토리 (기본: <root>/04_figure)")
    args = parser.parse_args()

    fu.setup_figure_style()

    result_dir = (Path(args.result_dir).resolve()
                  if args.result_dir else fu.get_result_dir("context_ablation"))
    out_dir = Path(args.out_dir).resolve() if args.out_dir else fu.get_figure_dir()

    print(f"[fig3] result_dir = {result_dir}")
    print(f"[fig3] out_dir    = {out_dir}")

    # summary 우선, 없으면 raw results 로 대체
    summary = load_csv(result_dir, "ablation_summary.csv")
    raw = load_csv(result_dir, "ablation_results.csv")

    acc_df = summary if summary is not None else raw  # Panel A
    time_df = raw if raw is not None else summary       # Panel B (timing은 raw)

    if acc_df is None and time_df is None:
        draw_placeholder(out_dir)
        return

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))

    if acc_df is not None and "MAE" in acc_df.columns:
        draw_panel_a(ax_a, acc_df, _present_context_lens(acc_df))
    else:
        ax_a.axis("off")
        ax_a.text(0.5, 0.5, "MAE data unavailable", ha="center", va="center",
                  color="#b2182b", transform=ax_a.transAxes)

    if time_df is not None:
        draw_panel_b(ax_b, time_df, _present_context_lens(time_df))
    else:
        ax_b.axis("off")
        ax_b.text(0.5, 0.5, "Timing data unavailable", ha="center",
                  va="center", color="#b2182b", transform=ax_b.transAxes)

    fig.suptitle(
        "Context-length ablation: accuracy and inference cost",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fu.save_figure(fig, out_dir, "Fig3_context_ablation")
    plt.close(fig)


if __name__ == "__main__":
    main()

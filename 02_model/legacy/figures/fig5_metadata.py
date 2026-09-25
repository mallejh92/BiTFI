"""
Fig. 5 — 온실 메타데이터 기반 정확도 비교

단동(Single-span) vs 연동(Multi-span) 그리고 온실 면적 구간(3단계)에 따른
imputation 정확도(nRMSE) 변화를 비교한다.

Layout: 1행 × 2열
  (a) 온실 유형별 nRMSE 비교 (Single-span vs Multi-span)
  (b) 온실 면적 구간별 nRMSE 비교 (Small / Medium / Large)

각 막대: 해당 그룹 내 온실별 평균 nRMSE 의 그룹 평균.
원 점(scatter): 개별 온실의 평균 nRMSE.

실행:
  python fig5_metadata.py
  python fig5_metadata.py --metric RMSE
  python fig5_metadata.py --out-dir /path/to/output
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_LEGACY_DIR = _HERE.parent
_MODEL_DIR = _LEGACY_DIR.parent
_FIGURES_DIR = _MODEL_DIR / "figures"
for _p in (_HERE, _MODEL_DIR, _FIGURES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fig_utils as fu


# ─── 면적 구간 정의 ──────────────────────────────────────────────────────────
AREA_BINS   = [0, 2000, 6000, float("inf")]
AREA_LABELS = ["Small\n(<2,000 m²)", "Medium\n(2,000–6,000 m²)", "Large\n(>6,000 m²)"]

TYPE_ORDER = ["Single-span", "Multi-span"]
TYPE_COLORS = {
    "Single-span": "#56B4E9",
    "Multi-span":  "#E69F00",
}

AREA_COLORS = {
    AREA_LABELS[0]: "#009E73",
    AREA_LABELS[1]: "#CC79A7",
    AREA_LABELS[2]: "#D55E00",
}


# ─── 데이터 준비 ─────────────────────────────────────────────────────────────
def load_and_merge(metric: str = "nRMSE") -> pd.DataFrame:
    data_dir   = fu.get_data_dir()
    result_csv = fu.get_result_dir("experiment_v2") / "results.csv"

    if not result_csv.exists():
        raise FileNotFoundError(f"결과 파일 없음: {result_csv}")

    meta = pd.read_excel(data_dir / "meta_data.xlsx")[
        ["name", "type", "area (m2)"]
    ].rename(columns={"name": "greenhouse_id", "type": "gh_type", "area (m2)": "area_m2"})

    df = pd.read_csv(result_csv)

    # 온실별·모델별 평균 metric
    gh_model = (
        df.groupby(["greenhouse_id", "model"])[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "value"})
    )

    merged = gh_model.merge(meta, on="greenhouse_id", how="inner")
    merged["area_bin"] = pd.cut(
        merged["area_m2"],
        bins=AREA_BINS,
        labels=AREA_LABELS,
    )
    return merged


# ─── 그리기 헬퍼 ─────────────────────────────────────────────────────────────
def _grouped_bars(
    ax,
    data: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    group_colors: dict[str, str],
    model_order: list[str],
    metric_label: str,
    panel_label: str,
) -> None:
    """model_order × group_order 의 묶음 막대 그래프."""
    n_models = len(model_order)
    n_groups = len(group_order)
    bar_w    = 0.7 / n_groups
    x_base   = np.arange(n_models)

    for g_idx, grp in enumerate(group_order):
        sub = data[data[group_col] == grp]
        means = []
        for m in model_order:
            vals = sub[sub["model"] == m]["value"].values
            means.append(vals.mean() if len(vals) > 0 else np.nan)

        offset = (g_idx - (n_groups - 1) / 2) * bar_w
        bars = ax.bar(
            x_base + offset, means,
            width=bar_w * 0.9,
            color=group_colors[grp],
            label=grp.replace("\n", " "),
            alpha=0.85,
            zorder=3,
        )

        # 개별 온실 값 overlay (점)
        for m_idx, m in enumerate(model_order):
            vals = sub[sub["model"] == m]["value"].values
            jitter = np.linspace(-bar_w * 0.2, bar_w * 0.2, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(
                x_base[m_idx] + offset + np.array(jitter),
                vals,
                color="white",
                edgecolors=group_colors[grp],
                s=28, zorder=5, linewidths=1.0,
            )

    model_display = [
        fu.MODEL_DISPLAY.get(m, m) for m in model_order
    ]
    ax.set_xticks(x_base)
    ax.set_xticklabels(model_display, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel(metric_label, fontsize=10)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.text(0.02, 0.98, f"({panel_label})",
            transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top")


# ─── 메인 그리기 ─────────────────────────────────────────────────────────────
def draw(merged: pd.DataFrame, metric: str, out_dir: Path) -> None:
    fu.setup_figure_style()

    model_order = [m for m in fu.MODEL_ORDER if m in merged["model"].unique()]
    metric_label = {"nRMSE": "nRMSE", "RMSE": "RMSE", "MAE": "MAE"}.get(metric, metric)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), squeeze=False)

    # ── (a) 온실 유형 비교 ──
    type_order = [t for t in TYPE_ORDER if t in merged["gh_type"].unique()]
    _grouped_bars(
        ax=axes[0][0],
        data=merged,
        group_col="gh_type",
        group_order=type_order,
        group_colors=TYPE_COLORS,
        model_order=model_order,
        metric_label=metric_label,
        panel_label="a",
    )
    axes[0][0].set_title("Greenhouse Type", fontsize=12, fontweight="bold")

    legend_handles_type = [
        Patch(facecolor=TYPE_COLORS[t], alpha=0.85, label=t)
        for t in type_order
    ]
    axes[0][0].legend(
        handles=legend_handles_type,
        fontsize=9, loc="upper right", frameon=True,
    )

    # ── (b) 면적 구간 비교 ──
    area_order = [a for a in AREA_LABELS if a in merged["area_bin"].values]
    _grouped_bars(
        ax=axes[0][1],
        data=merged,
        group_col="area_bin",
        group_order=area_order,
        group_colors=AREA_COLORS,
        model_order=model_order,
        metric_label=metric_label,
        panel_label="b",
    )
    axes[0][1].set_title("Greenhouse Area", fontsize=12, fontweight="bold")

    legend_handles_area = [
        Patch(facecolor=AREA_COLORS[a], alpha=0.85, label=a.replace("\n", " "))
        for a in area_order
    ]
    axes[0][1].legend(
        handles=legend_handles_area,
        fontsize=9, loc="upper right", frameon=True,
    )

    fig.suptitle(
        f"Imputation Accuracy by Greenhouse Metadata  ({metric_label})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    fname = f"Fig5_metadata_{metric}"
    fu.save_figure(fig, out_dir, fname)
    plt.close(fig)


# ─── 메인 ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Fig 5 — metadata-based accuracy comparison")
    parser.add_argument("--metric",  default="nRMSE",
                        choices=["nRMSE", "RMSE", "MAE"],
                        help="비교에 사용할 정확도 지표")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    fu.setup_figure_style()

    out_dir = Path(args.out_dir).resolve() if args.out_dir else fu.get_figure_dir()

    print(f"[fig5] 데이터 로딩 (metric={args.metric}) ...")
    merged = load_and_merge(args.metric)

    gh_summary = (
        merged[["greenhouse_id", "gh_type", "area_m2", "area_bin"]]
        .drop_duplicates()
        .sort_values("area_m2")
    )
    print("[fig5] 대상 온실:")
    for _, row in gh_summary.iterrows():
        print(f"  {row['greenhouse_id']:20s}  {row['gh_type']:12s}  "
              f"{row['area_m2']:8.0f} m²  → {str(row['area_bin'])}")

    draw(merged, args.metric, out_dir)
    print("[fig5] 완료")


if __name__ == "__main__":
    main()

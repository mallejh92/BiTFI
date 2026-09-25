"""
F5_stratified.py
Figure 5 — 계절 × 시간대 층화 성능.

입력 : 03_result/comparison/results.csv
        (group_type ∈ {'season','time_of_day','season_tod'})
출력 : 03_result/comparison/figures/F5_stratified.{pdf,png}

레이아웃 (1×3):
  (a) 모델 × 계절 NMAE 막대 그룹
  (b) season_tod 히트맵 — 제안기법(CAFI) vs 최강 baseline의 ΔNMAE
       (음수 = CAFI가 더 좋음, 파랑)
  (c) 환경 변동성(모델 독립적 동적 난이도) vs NMAE 조밀 산점도 — CAFI vs 최강 baseline
       점 = (온실 × 변수) 단위(10×5=50). 변동성 = 정규화 변수의 시간당 1차차분 std.
       최강 baseline은 전체 NMAE 최소인 TimesFM-2.5로 고정한다.
       복원 난이도가 시설 규모가 아니라 시계열 변동성(제어 이벤트 강도)으로
       결정됨을 보인다(원자료 필요; 없으면 안내문으로 대체).

실행:
  python F5_stratified.py
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
if str(_HERE.parent) not in sys.path:            # 02_model (preprocessing/data_split 접근)
    sys.path.insert(0, str(_HERE.parent))
import fig_utils as fu

_SEASON_ORDER = ["spring", "summer", "fall", "winter"]
_SEASON_DISPLAY = {"spring": "Spring", "summer": "Summer", "fall": "Fall", "winter": "Winter"}


def _season_panel(ax, df: pd.DataFrame, models: list[str]) -> None:
    s = df[df["group_type"] == "season"]
    seasons = [x for x in _SEASON_ORDER if x in s["group_value"].unique()]
    x = np.arange(len(seasons))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        ys = [fu.wavg_nmae(s[(s["model"] == m) & (s["group_value"] == sv)]) for sv in seasons]
        ax.bar(x + i * width, ys, width, label=fu.cmp_model_display(m),
               color=fu.cmp_model_color(m), edgecolor="black", linewidth=0.4)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([_SEASON_DISPLAY.get(sv, sv.capitalize()) for sv in seasons])
    ax.set_ylabel("NMAE")
    ax.set_title("(a) NMAE by season")
    ax.legend(ncol=2, fontsize=7)


def _season_tod_delta(ax, df: pd.DataFrame, models: list[str]) -> None:
    st = df[df["group_type"] == "season_tod"]
    if st.empty or "CAFI" not in models:
        ax.axis("off")
        ax.text(0.5, 0.5, "season_tod / CAFI 데이터 없음", ha="center", va="center")
        return
    buckets = sorted(st["group_value"].unique())
    baselines = [m for m in models if m != "CAFI"]
    # 각 버킷에서 CAFI NMAE - (baseline 중 최소 NMAE)
    cafi = np.array([fu.wavg_nmae(st[(st["model"] == "CAFI") & (st["group_value"] == b)]) for b in buckets])
    best_base = np.array([
        np.nanmin([fu.wavg_nmae(st[(st["model"] == bm) & (st["group_value"] == b)]) for bm in baselines])
        for b in buckets
    ])
    delta = cafi - best_base
    vmax = np.nanmax(np.abs(delta)) if np.isfinite(delta).any() else 1.0
    im = ax.imshow(delta.reshape(-1, 1), aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    def _fmt_bucket(b: str) -> str:
        parts = b.split("_")
        parts[0] = _SEASON_DISPLAY.get(parts[0], parts[0].capitalize())
        return "_".join(parts)
    ax.set_yticks(range(len(buckets)))
    ax.set_yticklabels([_fmt_bucket(b) for b in buckets], fontsize=8)
    ax.set_xticks([])
    ax.set_title("(b) CAFI − best baseline\n(ΔNMAE; blue = CAFI better)")
    for i, d in enumerate(delta):
        if np.isfinite(d):
            ax.text(0, i, f"{d:+.3f}", ha="center", va="center", fontsize=8,
                    color="black")
    plt.colorbar(im, ax=ax, fraction=0.12, pad=0.04, label="ΔNMAE")


_BASELINE_REF = "TimesFM2.5"   # 전체 NMAE가 가장 낮은 비교 모델(최강 baseline)


def _volatility_table(result_dir: Path | None = None) -> pd.DataFrame:
    """(온실 × 변수)별 환경 변동성 = 정규화 변수의 시간당 1차차분 std.

    시설 규모·작물 같은 정적 속성이 아니라 시계열의 급변/이벤트성(제어 강도)을
    모델과 무관하게 정량화한다. 원자료(01_data/*.xlsx, gitignore)가 없으면 빈 DF.
    """
    try:
        from data_split import load_split
        from preprocessing import preprocess_file, TARGET_COLS
    except Exception:
        return pd.DataFrame(columns=["greenhouse", "variable", "vol"])
    rd = result_dir if result_dir is not None else fu.get_comparison_dir()
    try:
        test_paths = [Path(p) for p in load_split(rd)["test"]]
    except Exception:
        return pd.DataFrame(columns=["greenhouse", "variable", "vol"])
    targets = list(TARGET_COLS.values())
    rows = []
    for fp in test_paths:
        try:
            res = preprocess_file(fp, include_covariates=True)
        except Exception:
            res = None
        if not res:
            continue
        data = res["data"]  # min–max 정규화됨 → 변수 간 스케일 통일
        if any(c not in data.columns for c in targets):
            continue
        for v in targets:
            rows.append({"greenhouse": res["name"], "variable": v,
                         "vol": float(data[v].diff().std())})
    return pd.DataFrame(rows)


# 변수별 색상(어떤 변수가 변동성이 큰지 시각적으로 구분).
_VAR_COLORS = {"Tin": "#d62728", "Tout": "#1f77b4", "RH": "#2ca02c",
               "CO2": "#9467bd", "Rad": "#ff7f0e"}
_VAR_LABELS = {"Tin": r"$T_{in}$", "Tout": r"$T_{out}$", "RH": "RH",
               "CO2": r"CO$_2$", "Rad": "Radiation"}


def _volatility_panel(ax, df: pd.DataFrame, result_dir: Path | None) -> None:
    """(c) (온실×변수)별 환경 변동성 vs NMAE 조밀 산점도.

    점 색 = 변수(어떤 변수의 변동성이 큰지), CAFI는 원(●)/TimesFM-2.5는 x(×)로 구분.
    """
    vt = _volatility_table(result_dir)
    allr = df[df["group_type"] == "all"]
    if vt.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "원자료 없음: 변동성 계산 불가", ha="center", va="center")
        return
    xs, y_cafi, y_ref, cols = [], [], [], []
    for _, row in vt.iterrows():
        sub = allr[(allr["greenhouse"] == row["greenhouse"]) & (allr["variable"] == row["variable"])]
        c = fu.wavg_nmae(sub[sub["model"] == "CAFI"])
        r = fu.wavg_nmae(sub[sub["model"] == _BASELINE_REF])
        if np.isfinite(c) and np.isfinite(r):
            xs.append(row["vol"]); y_cafi.append(c); y_ref.append(r)
            cols.append(_VAR_COLORS.get(row["variable"], "0.5"))
    xs = np.asarray(xs); y_cafi = np.asarray(y_cafi); y_ref = np.asarray(y_ref)
    # 변수별 색: CAFI=원, TimesFM-2.5=×
    ax.scatter(xs, y_ref, s=34, c=cols, marker="x", linewidth=1.1, alpha=0.9, zorder=3)
    ax.scatter(xs, y_cafi, s=30, c=cols, marker="o", edgecolor="black", linewidth=0.3,
               alpha=0.9, zorder=4)
    if len(xs) >= 4:
        xg = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(xg, np.polyval(np.polyfit(xs, y_ref, 1), xg), color="0.45",
                linestyle="--", linewidth=1.3, zorder=1)
        ax.plot(xg, np.polyval(np.polyfit(xs, y_cafi, 1), xg), color="black",
                linestyle="--", linewidth=1.4, zorder=2)
        # Spearman 순위 상관(변동성 측도 선택·꼬리에 강건) = 순위값의 Pearson 상관.
        rho = np.corrcoef(pd.Series(xs).rank(), pd.Series(y_cafi).rank())[0, 1]
        ax.text(0.04, 0.96, rf"CAFI: Spearman $\rho$={rho:.2f} (n={len(xs)})",
                transform=ax.transAxes, va="top", ha="left", fontsize=8, fontweight="bold")
    # 범례 1: 변수 색상
    from matplotlib.lines import Line2D
    var_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=_VAR_COLORS[v],
                          markeredgecolor="black", markersize=7, label=_VAR_LABELS[v])
                   for v in _VAR_COLORS if v in vt["variable"].unique()]
    leg1 = ax.legend(handles=var_handles, fontsize=7, loc="lower right",
                     title="Variable", title_fontsize=7, ncol=2)
    ax.add_artist(leg1)
    # 범례 2: 모델(마커) 구분
    mdl_handles = [
        Line2D([0], [0], marker="o", color="0.35", markerfacecolor="0.7",
               markersize=7, linestyle="none", label="CAFI"),
        Line2D([0], [0], marker="x", color="0.35", markersize=7, linestyle="none",
               label=f"{fu.cmp_model_display(_BASELINE_REF)} (best baseline)"),
    ]
    ax.legend(handles=mdl_handles, fontsize=7, loc="upper left",
              bbox_to_anchor=(0.0, 0.90))
    ax.set_xlabel("Environmental volatility\n(SD of hourly differences, per variable)")
    ax.set_ylabel("NMAE")
    ax.set_title("(c) Dynamic difficulty vs accuracy")


def generate(result_dir: Path | None = None) -> None:
    df = fu.load_results(result_dir)
    models = fu.ordered_models(df["model"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2),
                             gridspec_kw={"width_ratios": [2.0, 0.9, 1.5]})
    _season_panel(axes[0], df, models)
    _season_tod_delta(axes[1], df, models)
    _volatility_panel(axes[2], df, result_dir)
    fig.tight_layout()
    out_dir = (Path(result_dir) if result_dir else fu.get_comparison_dir()) / "figures"
    fu.save_figure(fig, out_dir, "F5_stratified")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="F5 계절×시간대 층화")
    p.add_argument("--result-dir", type=str, default=str(fu.get_comparison_dir()))
    args = p.parse_args()
    generate(Path(args.result_dir))

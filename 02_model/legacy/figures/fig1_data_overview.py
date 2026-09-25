"""
Fig. 1 — 데이터 개요
온실 환경 시계열 데이터의 구조, 통계, 시각적 예시를 보여준다.

Layout (4-panel figure, 14×12 inches):
  Panel A (top-left): Meta data table - greenhouse IDs, number of data points, time period, missing rate per variable
  Panel B (top-right): Correlation heatmap of the 5 variables (pooled across all greenhouses)
  Panel C (middle): Multi-variable time series for one representative greenhouse (~30 days of raw data)
  Panel D (bottom): Missing data pattern visualization - greenhouse (rows) × variable, colored=observed white=missing

실행: python fig1_data_overview.py [--data-dir ../../01_data] [--out-dir ../../04_figure]
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
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_HERE = Path(__file__).resolve().parent
_LEGACY_DIR = _HERE.parent
_MODEL_DIR = _LEGACY_DIR.parent
_FIGURES_DIR = _MODEL_DIR / "figures"
for _p in (_HERE, _MODEL_DIR, _FIGURES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fig_utils as fu  # noqa: E402
from preprocessing import preprocess_directory  # noqa: E402


# 대표 온실 (파일 크기 최대). 없으면 가장 큰 데이터셋으로 자동 대체.
REPRESENTATIVE_GH = "PF_0025101_01"


# ---------------------------------------------------------------------------
# 데이터 로드 / 통계
# ---------------------------------------------------------------------------
def load_meta_table(data_dir: Path, results: list[dict]) -> pd.DataFrame:
    """그림 Panel A 용 메타 테이블을 만든다.

    meta_data.xlsx 가 있으면 참고하되, 신뢰 가능한 통계(행 수, 기간, freq,
    변수별 결측률)는 전처리 결과에서 직접 계산한다.
    """
    rows = []
    for r in results:
        df_raw = r["data_raw"]
        idx = df_raw.index
        try:
            start = pd.Timestamp(idx.min()).strftime("%Y-%m-%d")
            end = pd.Timestamp(idx.max()).strftime("%Y-%m-%d")
            period = f"{start}\n~{end}"
        except Exception:
            period = "n/a"

        mr = r.get("missing_rate", {}) or {}
        row = {
            "Greenhouse ID": r["name"],
            "Data Points": f"{len(df_raw):,}",
            "Time Period": period,
            "Freq": r.get("freq", "?"),
        }
        for v in fu.VARIABLES:
            rate = mr.get(v)
            if rate is None and v in df_raw.columns:
                rate = float(df_raw[v].isna().mean())
            row[f"{v} Missing%"] = f"{(rate or 0.0) * 100:.1f}"
        rows.append(row)

    return pd.DataFrame(rows)


def pooled_correlation(results: list[dict]) -> pd.DataFrame:
    """모든 온실의 data_raw 를 concat 하여 5변수 상관행렬을 계산한다."""
    frames = []
    for r in results:
        df = r["data_raw"]
        cols = [c for c in fu.VARIABLES if c in df.columns]
        if cols:
            frames.append(df[cols])
    if not frames:
        return pd.DataFrame(
            np.eye(len(fu.VARIABLES)),
            index=fu.VARIABLES,
            columns=fu.VARIABLES,
        )
    pooled = pd.concat(frames, axis=0, ignore_index=True)
    pooled = pooled.reindex(columns=fu.VARIABLES)
    return pooled.corr()


def pick_representative(results: list[dict]) -> dict:
    """대표 온실 dict 를 선택한다 (지정 ID 우선, 없으면 최대 행 수)."""
    for r in results:
        if r["name"] == REPRESENTATIVE_GH:
            return r
    return max(results, key=lambda r: len(r["data_raw"]))


# ---------------------------------------------------------------------------
# 패널 그리기
# ---------------------------------------------------------------------------
def _panel_label(ax, text: str) -> None:
    ax.text(
        -0.02, 1.05, text, transform=ax.transAxes,
        fontsize=14, fontweight="bold", va="bottom", ha="left",
    )


def draw_panel_a(ax, meta: pd.DataFrame) -> None:
    """Panel A: 메타 데이터 테이블."""
    ax.axis("off")
    _panel_label(ax, "(A)")
    ax.set_title("Greenhouse dataset summary", pad=18, loc="left")

    # 헤더 단축 (Missing% 열을 변수명만 표기 + 단위 헤더 행으로 축약)
    display_cols = []
    for c in meta.columns:
        display_cols.append(c.replace(" Missing%", "\nmiss.%").replace(" ", "\n"))

    cell_text = meta.values.tolist()
    table = ax.table(
        cellText=cell_text,
        colLabels=display_cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.6)

    n_rows, n_cols = meta.shape
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:  # 헤더
            cell.set_facecolor("#4C4C4C")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(cell.get_height() * 1.4)
        else:
            # 행 교차 색
            cell.set_facecolor("#f2f2f2" if (row % 2 == 0) else "#ffffff")
            # 결측률 열은 값에 따라 강조 (>30% 붉게)
            if col >= 4:
                try:
                    val = float(cell_text[row - 1][col])
                    if val >= 30:
                        cell.set_text_props(color="#b2182b", fontweight="bold")
                    elif val >= 10:
                        cell.set_text_props(color="#d6604d")
                except (ValueError, IndexError):
                    pass


def draw_panel_b(ax, corr: pd.DataFrame) -> None:
    """Panel B: 5변수 상관 히트맵 (imshow 기반, seaborn 불필요)."""
    _panel_label(ax, "(B)")
    ax.set_title("Variable correlation (pooled)", loc="left")

    mat = corr.reindex(index=fu.VARIABLES, columns=fu.VARIABLES).values
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")

    ax.set_xticks(range(len(fu.VARIABLES)))
    ax.set_yticks(range(len(fu.VARIABLES)))
    ax.set_xticklabels(fu.VARIABLES, rotation=45, ha="right")
    ax.set_yticklabels(fu.VARIABLES)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 각 셀에 상관계수 주석
    for i in range(len(fu.VARIABLES)):
        for j in range(len(fu.VARIABLES)):
            val = mat[i, j]
            if np.isnan(val):
                txt = "—"
                color = "#777777"
            else:
                txt = f"{val:.2f}"
                color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8.5, color=color)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=10)
    cbar.ax.tick_params(labelsize=8)


def draw_panel_c(axes, rep: dict, n_days: int = 30) -> None:
    """Panel C: 대표 온실의 30일 다변량 원시 시계열 (변수별 stacked subplot)."""
    df = rep["data_raw"].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # 가능하면 가장 결측이 적은 30일 창을 고르기 위해 시작점부터 window 사용
    if isinstance(df.index, pd.DatetimeIndex) and len(df):
        start = df.index.min()
        window = df.loc[start: start + pd.Timedelta(days=n_days)]
        if len(window) < 10:  # window 가 비정상적으로 짧으면 앞부분 N행 사용
            window = df.iloc[: min(len(df), n_days * 24)]
    else:
        window = df.iloc[: min(len(df), n_days * 24)]

    _panel_label(axes[0], "(C)")
    axes[0].set_title(
        f"Representative greenhouse time series — {rep['name']} "
        f"(first ~{n_days} days)",
        loc="left",
    )

    for ax, v in zip(axes, fu.VARIABLES):
        if v in window.columns:
            ax.plot(
                window.index, window[v].values,
                color=fu.VARIABLE_COLORS[v], linewidth=0.9,
            )
        ax.set_ylabel(
            fu.VARIABLE_LABELS[v].split("(")[0].strip() + f"\n({fu.VARIABLE_UNITS[v]})",
            fontsize=9,
        )
        ax.tick_params(labelsize=8)
        ax.margins(x=0.01)
        # 마지막 subplot 외에는 x 라벨 숨김
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel("Date", fontsize=10)


def draw_panel_d(ax, results: list[dict]) -> None:
    """Panel D: 온실(행) × 변수(블록)별 관측/결측 패턴.

    각 온실 행 안에서 5개 변수를 세로로 얇게 쌓아, 시간축(가로)을 따라
    관측=색, 결측=흰색 으로 표시한다.
    """
    _panel_label(ax, "(D)")
    ax.set_title("Missing-data pattern (colored = observed, white = missing)",
                 loc="left")

    n_gh = len(results)
    n_var = len(fu.VARIABLES)
    n_bins = 400  # 시간축 다운샘플 해상도

    # 각 (온실, 변수) 마다 한 줄. 온실 사이에 작은 간격.
    sub_h = 1.0 / n_var
    yticks = []
    yticklabels = []

    for gi, r in enumerate(results):
        df = r["data_raw"]
        row_base = n_gh - 1 - gi  # 위에서부터 첫 온실
        for vi, v in enumerate(fu.VARIABLES):
            y0 = row_base + vi * sub_h
            color = fu.VARIABLE_COLORS[v]
            if v in df.columns:
                series = df[v].values
                # n_bins 구간으로 나눠 관측 여부 요약 (한 구간이라도 값 있으면 관측)
                idx = np.linspace(0, len(series), n_bins + 1).astype(int)
                observed = np.zeros(n_bins, dtype=bool)
                for b in range(n_bins):
                    s, e = idx[b], idx[b + 1]
                    if e > s:
                        observed[b] = np.any(~np.isnan(series[s:e]))
                # 관측 구간을 broken_barh 로 그림
                spans = []
                b = 0
                while b < n_bins:
                    if observed[b]:
                        start = b
                        while b < n_bins and observed[b]:
                            b += 1
                        spans.append((start, b - start))
                    else:
                        b += 1
                widths = [(s / n_bins, w / n_bins) for s, w in spans]
                if widths:
                    ax.broken_barh(widths, (y0, sub_h * 0.9),
                                   facecolors=color, edgecolors="none")
            # 변수 행의 중앙 y 위치를 첫 온실에서만 라벨용으로 기록 안함
        # 온실 라벨: 행 블록 중앙
        yticks.append(row_base + 0.5)
        yticklabels.append(r["name"])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_gh)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=8)
    ax.set_xlabel("Time span (relative position within each record)", fontsize=10)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    # 온실 경계선
    for gi in range(1, n_gh):
        ax.axhline(gi, color="#dddddd", linewidth=0.6)

    legend_handles = [
        Patch(facecolor=fu.VARIABLE_COLORS[v], label=v) for v in fu.VARIABLES
    ]
    ax.legend(
        handles=legend_handles, ncol=5, fontsize=8,
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        title="Within each greenhouse row, variables are stacked top→bottom: "
              + " / ".join(fu.VARIABLES),
        title_fontsize=8,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_figure(results: list[dict]):
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(
        4, 2, figure=fig,
        height_ratios=[1.1, 1.6, 0.05, 1.4],  # row2 = 5변수 inner grid 용
        hspace=0.55, wspace=0.28,
    )

    # Panel A (top-left)
    ax_a = fig.add_subplot(gs[0, 0])
    meta = load_meta_table(fu.get_data_dir(), results)
    draw_panel_a(ax_a, meta)

    # Panel B (top-right)
    ax_b = fig.add_subplot(gs[0, 1])
    corr = pooled_correlation(results)
    draw_panel_b(ax_b, corr)

    # Panel C (middle, full width) — 5 stacked subplots via nested gridspec
    rep = pick_representative(results)
    gs_c = gs[1, :].subgridspec(len(fu.VARIABLES), 1, hspace=0.12)
    axes_c = [fig.add_subplot(gs_c[i, 0]) for i in range(len(fu.VARIABLES))]
    draw_panel_c(axes_c, rep)

    # Panel D (bottom, full width)
    ax_d = fig.add_subplot(gs[3, :])
    draw_panel_d(ax_d, results)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Fig 1 — data overview")
    parser.add_argument("--data-dir", default=None,
                        help="원본 데이터 디렉토리 (기본: <root>/01_data)")
    parser.add_argument("--out-dir", default=None,
                        help="그림 출력 디렉토리 (기본: <root>/04_figure)")
    args = parser.parse_args()

    fu.setup_figure_style()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else fu.get_data_dir()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else fu.get_figure_dir()

    print(f"[fig1] data_dir = {data_dir}")
    print(f"[fig1] out_dir  = {out_dir}")

    results = preprocess_directory(data_dir)
    if not results:
        # 데이터가 없으면 placeholder 생성
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.axis("off")
        ax.text(0.5, 0.5, "Data not available\n(no PF_*.xlsx found)",
                ha="center", va="center", fontsize=20, color="#b2182b")
        fu.save_figure(fig, out_dir, "Fig1_data_overview")
        plt.close(fig)
        return

    fig = build_figure(results)
    fu.save_figure(fig, out_dir, "Fig1_data_overview")
    plt.close(fig)


if __name__ == "__main__":
    main()

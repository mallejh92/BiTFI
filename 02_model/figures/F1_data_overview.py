"""
F1_data_overview.py
Figure 1 — 데이터셋 개요 (출판용).

입력 : 01_data/*.xlsx, 03_result/comparison/split.json
출력 : 03_result/comparison/figures/F1_data_overview.{pdf,png}

레이아웃 (gridspec 3행):
  (a) 온실별 데이터 수집 기간 Gantt — train/test 색 + 결측 구간 음영, 월별 tick,
      y축은 Train NN / Test NN 로 표기(원시 코드 대신).
  (b) 변수별 값 분포 — seaborn violinplot, 내부에 Q1·중앙값·Q3 점선(quartile).
  (c) 변수별 결측률 (train vs test).
  (d) 변수 간 상관 heatmap — 온실별 Pearson 상관의 평균(covariate 포함).
      → CAFI가 활용하는 변수 간 상관을 직접 보여줌.

실행:
  python F1_data_overview.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
for _p in (_HERE, _MODEL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gpu_utils  # noqa: F401
import fig_utils as fu
from preprocessing import preprocess_file, TARGET_COLS

TARGET_VARS = list(TARGET_COLS.values())
COV_VARS = ["WindSpeed", "RadIn"]
ALL_VARS = TARGET_VARS + COV_VARS
_ROLE_COLOR = {"train": "#9ecae1", "test": "#fc9272"}
_MISS_COLOR = "#444444"


def _runs(mask: np.ndarray, idx: pd.DatetimeIndex):
    """boolean mask의 연속 True 구간을 (start_dt, end_dt) 리스트로 반환."""
    out = []
    n = len(mask); in_run = False; st = 0
    for i in range(n):
        if mask[i] and not in_run:
            st = i; in_run = True
        elif not mask[i] and in_run:
            out.append((idx[st], idx[i - 1])); in_run = False
    if in_run:
        out.append((idx[st], idx[n - 1]))
    return out


def _collect(split: dict, sample_per_gh: int = 3000):
    recs = []          # {name, role, start, end, miss_intervals}
    miss_rows = []
    dist = {v: [] for v in TARGET_VARS}
    corr_sum = pd.DataFrame(0.0, index=ALL_VARS, columns=ALL_VARS)
    corr_cnt = pd.DataFrame(0.0, index=ALL_VARS, columns=ALL_VARS)
    rng = np.random.RandomState(0)

    for role in ("train", "test"):
        for p in split.get(role, []):
            res = preprocess_file(Path(p), include_covariates=True)
            if res is None:
                continue
            name = res["name"]; raw = res["data_raw"]; norm = res["data"]
            idx = pd.DatetimeIndex(norm.index)
            tcols = [c for c in TARGET_VARS if c in norm.columns]
            miss_any = norm[tcols].isna().any(axis=1).values
            recs.append({"name": name, "role": role, "start": idx.min(), "end": idx.max(),
                         "miss": _runs(miss_any, idx)})
            mr = res["missing_rate"]
            miss_rows.append({"greenhouse": name, "role": role,
                              **{v: float(mr.get(v, np.nan)) for v in TARGET_VARS}})
            for v in tcols:
                vals = raw[v].dropna().values
                if len(vals):
                    take = vals if len(vals) <= sample_per_gh else vals[rng.choice(len(vals), sample_per_gh, replace=False)]
                    dist[v].append(take)
            # 온실별 Pearson 상관 (정규화는 선형이라 raw 상관과 동일)
            cols = [c for c in ALL_VARS if c in norm.columns]
            c = norm[cols].corr()
            for a in cols:
                for b in cols:
                    if np.isfinite(c.loc[a, b]):
                        corr_sum.loc[a, b] += c.loc[a, b]; corr_cnt.loc[a, b] += 1
    corr_mean = (corr_sum / corr_cnt.replace(0, np.nan))
    return recs, pd.DataFrame(miss_rows), dist, corr_mean


def _gantt(ax, recs):
    train = sorted([r for r in recs if r["role"] == "train"], key=lambda r: r["start"])
    test = sorted([r for r in recs if r["role"] == "test"], key=lambda r: r["start"])
    ordered = train + test
    labels = [f"{i+1:02d}" for i in range(len(train))] + \
             [f"{i+1:02d}" for i in range(len(test))]
    g_min = min(r["start"] for r in ordered); g_max = max(r["end"] for r in ordered)

    for k, r in enumerate(ordered):
        s, e = mdates.date2num(r["start"]), mdates.date2num(r["end"])
        ax.barh(k, e - s, left=s, height=0.7, color=_ROLE_COLOR[r["role"]],
                alpha=0.7, edgecolor="none", zorder=2)

    ax.set_yticks(range(len(ordered))); ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(mdates.date2num(g_min), mdates.date2num(g_max))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(45); lab.set_fontsize(7); lab.set_ha("right")
    ax.set_title(f"(a) Data coverage by greenhouse  (train={len(train)}, test={len(test)})",
                 fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_ROLE_COLOR["train"], alpha=0.7),
               plt.Rectangle((0, 0), 1, 1, color=_ROLE_COLOR["test"], alpha=0.7)]
    ax.legend(handles, ["train", "test"], fontsize=8, loc="lower right", ncol=2)


def _violins(axes, dist):
    for j, v in enumerate(TARGET_VARS):
        ax = axes[j]
        data = np.concatenate(dist[v]) if dist[v] else np.array([0.0])
        sns.violinplot(y=data, ax=ax, color=fu.VARIABLE_COLORS.get(v, "#888"),
                       inner="quartile", cut=0, linewidth=1.0)
        ax.set_xticks([]); ax.set_xlabel(fu.VAR_SHORT_LABELS.get(v, v), fontsize=9)
        ax.set_ylabel(fu.VARIABLE_UNITS.get(v, ""), fontsize=8); ax.tick_params(labelsize=7)
    axes[0].figure.text(0.5, axes[0].get_position().y1 + 0.03,
                        "(b) Variable distributions (real units; dashed = Q1/median/Q3)",
                        ha="center", fontsize=11)


def _missing_bar(ax, miss_df):
    width = 0.38; x = np.arange(len(TARGET_VARS))
    for i, role in enumerate(("train", "test")):
        sub = miss_df[miss_df["role"] == role]
        ys = [sub[v].mean() * 100 if v in sub else np.nan for v in TARGET_VARS]
        ax.bar(x + (i - 0.5) * width, ys, width, label=role,
               color=_ROLE_COLOR[role], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([fu.VAR_SHORT_LABELS.get(v, v) for v in TARGET_VARS], fontsize=8)
    ax.set_ylabel("missing rate (%)"); ax.set_title("(c) Missing rate by variable", fontsize=11)
    ax.legend(fontsize=8)


def _corr_heatmap(ax, corr_mean):
    vars_present = [v for v in TARGET_VARS if v in corr_mean.index and corr_mean.loc[v].notna().any()]
    mat = corr_mean.loc[vars_present, vars_present].values.astype(float)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels_present = [fu.VAR_SHORT_LABELS.get(v, v) for v in vars_present]
    ax.set_xticks(range(len(vars_present))); ax.set_xticklabels(labels_present, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(vars_present))); ax.set_yticklabels(labels_present, fontsize=8)
    for i in range(len(vars_present)):
        for j in range(len(vars_present)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(mat[i, j]) > 0.6 else "black")
    ax.set_title("(d) Inter-variable correlation (mean over greenhouses)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")


def generate(result_dir: Path | None = None) -> None:
    rd = Path(result_dir) if result_dir else fu.get_comparison_dir()
    split = json.load(open(rd / "split.json"))
    recs, miss_df, dist, corr_mean = _collect(split)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.5, 1.0, 1.1], hspace=0.55, wspace=0.5)

    _gantt(fig.add_subplot(gs[0, :]), recs)
    _violins([fig.add_subplot(gs[1, j]) for j in range(5)], dist)
    _missing_bar(fig.add_subplot(gs[2, 0:2]), miss_df)
    _corr_heatmap(fig.add_subplot(gs[2, 2:4]), corr_mean)

    fig.suptitle("Greenhouse environmental time-series dataset overview", fontsize=14, y=0.99)
    fu.save_figure(fig, rd / "figures", "F1_data_overview")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="F1 데이터셋 개요")
    p.add_argument("--result-dir", type=str, default=str(fu.get_comparison_dir()))
    args = p.parse_args()
    generate(Path(args.result_dir))

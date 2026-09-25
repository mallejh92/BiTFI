"""
FS2_round_drift.py
Supplementary Figure S2 — CAFI 반복 refinement의 라운드별 정확도 변화.

입력 : 03_result/round_sweep/rounds_{chronos,tfm3}.csv  (round_sweep.py 출력)
출력 : 03_result/comparison/figures/FS2_round_drift.{pdf,png}

그림 : 패널 = 시나리오(B | C). x = 라운드(R0..R5), y = gap MAE (R0 대비 %).
       선 = 백본(Chronos-2, TimesFM-3.0), 밴드 = 온실×gap×repeat 단위 bootstrap 95% CI.
       각 선의 최소 라운드를 마커로 강조. R0=100% 수평선.

실행:
  python FS2_round_drift.py
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

_BACKBONES = {"chronos": ("CAFI (Chronos-2)", fu.cmp_model_color("CAFI")),
              "tfm3": ("CAFI (TimesFM-3.0)", fu.cmp_model_color("CAFI-TimesFM3"))}
_SCEN_TITLE = {"B": "(a) Scenario B — 3 internal sensors missing",
               "C": "(b) Scenario C — all 5 variables missing"}
_UNIT = ["greenhouse", "gap_length_h", "repeat"]


def _sweep_dir() -> Path:
    return fu.get_comparison_dir().parent / "round_sweep"


def _load() -> pd.DataFrame:
    frames = []
    for b in _BACKBONES:
        f = _sweep_dir() / f"rounds_{b}.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    if not frames:
        raise FileNotFoundError(f"round_sweep 결과 없음: {_sweep_dir()}")
    df = pd.concat(frames, ignore_index=True)
    return df[df["variable"] == "ALL"]


def _relative(df: pd.DataFrame) -> pd.DataFrame:
    """각 (backbone, scenario, unit)에서 R0 대비 % 로 변환."""
    key = ["backbone", "scenario"] + _UNIT
    r0 = df[df["round"] == 0].set_index(key)["mae_norm"]
    out = df.set_index(key).copy()
    out["rel"] = out["mae_norm"].div(r0.reindex(out.index)).mul(100.0).values
    return out.reset_index()


def _ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def generate(result_dir: Path | None = None) -> pd.DataFrame:
    df = _relative(_load())
    rounds = sorted(df["round"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    summary_rows = []

    for ax, scen in zip(axes, ["B", "C"]):
        sub = df[df["scenario"] == scen]
        for b, (label, color) in _BACKBONES.items():
            s = sub[sub["backbone"] == b]
            if s.empty:
                continue
            means, lo, hi = [], [], []
            for r in rounds:
                x = s.loc[s["round"] == r, "rel"].to_numpy()
                means.append(np.nanmean(x) if len(x) else np.nan)
                l, h = _ci(x); lo.append(l); hi.append(h)
            means = np.array(means)
            ax.fill_between(rounds, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(rounds, means, color=color, marker="o", ms=3.5, lw=1.8, label=label)
            k = int(np.nanargmin(means))
            ax.plot(rounds[k], means[k], marker="o", ms=9, mfc="none", mec=color, mew=1.8)
            abs_by_round = s.groupby("round")["mae_norm"].mean()
            for r in rounds:
                summary_rows.append(dict(scenario=scen, backbone=label, round=r,
                                         mae_norm=abs_by_round.get(r, np.nan),
                                         rel_pct=means[rounds.index(r)]))
        ax.axhline(100, color="0.55", lw=0.8, ls="--")
        ax.set_title(_SCEN_TITLE[scen], fontsize=9.5, loc="left")
        ax.set_xticks(rounds)
        ax.set_xticklabels([f"R{r}" for r in rounds])
        ax.set_xlabel("Refinement round (R0 = univariate initial estimate)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Gap MAE relative to R0 (%)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()

    out_dir = (result_dir or fu.get_comparison_dir()) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"FS2_round_drift.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out_dir / "FS2_round_drift_summary.csv", index=False, encoding="utf-8-sig")
    print(summ.pivot_table(index=["scenario", "backbone"], columns="round", values="rel_pct")
              .round(1).to_string())
    print(f"\n저장 → {out_dir / 'FS2_round_drift.pdf'}")
    return summ


if __name__ == "__main__":
    generate()

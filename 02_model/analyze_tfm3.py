"""
analyze_tfm3.py
TimesFM 3.0 추가 결과 분석 — 기존 9개 방법론 대비 head-to-head.

기존 results.csv(9모델)와 comparison_tfm3/results.csv(TimesFM3.0)를 합쳐
논문 Table 1/2/3과 동일한 집계(n_eval 가중 NMAE → 온실 평균 ± SD)로 비교한다.

실행:
  python analyze_tfm3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "figures"))
import fig_utils as fu  # noqa: E402

MAIN = _HERE / "../03_result/comparison/results.csv"
TFM3 = _HERE / "../03_result/comparison_tfm3/results.csv"
VAR  = _HERE / "../03_result/comparison_tfm3_variants/results.csv"
COV  = _HERE / "../03_result/comparison_tfm3_cov/results.csv"
CTF3 = _HERE / "../03_result/comparison_cafi_tfm3/results.csv"

KEY = ["greenhouse", "scenario", "masked_vars", "gap_length_h", "repeat",
       "group_type", "group_value", "variable"]


def load() -> pd.DataFrame:
    parts = [pd.read_csv(MAIN), pd.read_csv(TFM3)]
    for extra in (VAR, COV, CTF3):
        if extra.exists():
            parts.append(pd.read_csv(extra))
    df = pd.concat(parts, ignore_index=True)
    print(f"병합: {len(df):,}행 / {df.model.nunique()}개 모델")
    return df


def per_gh_nmae(df: pd.DataFrame, model: str) -> pd.Series:
    """온실별 n_eval 가중 NMAE (group_type='all' 기준)."""
    sub = df[(df.model == model) & (df.group_type == "all")]
    return sub.groupby("greenhouse").apply(fu.wavg_nmae, include_groups=False).dropna()


def table_overall(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        s = per_gh_nmae(df, m)
        rows.append({"model": m, "NMAE": s.mean(), "SD": s.std(), "n_gh": len(s)})
    t = pd.DataFrame(rows).sort_values("NMAE").reset_index(drop=True)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    return t


def table_by(df: pd.DataFrame, models: list[str], col: str,
             group_type: str = "all", order: list | None = None) -> pd.DataFrame:
    sub = df[(df.group_type == group_type) & (df.model.isin(models))]
    t = (sub.groupby(["model", col])
            .apply(fu.wavg_nmae, include_groups=False)
            .unstack(col))
    if order:
        t = t[[c for c in order if c in t.columns]]
    return t.reindex([m for m in models if m in t.index])


def paired_test(df: pd.DataFrame, m_a: str, m_b: str) -> None:
    """공통 평가셀에서 짝지은 NMAE 비교 (Wilcoxon signed-rank)."""
    from scipy.stats import wilcoxon
    sub = df[(df.group_type == "all") & (df.model.isin([m_a, m_b]))]
    piv = sub.pivot_table(index=KEY, columns="model", values="NMAE")
    piv = piv.dropna(subset=[m_a, m_b])
    if len(piv) == 0:
        print(f"  {m_a} vs {m_b}: 공통 셀 없음")
        return
    d = piv[m_a] - piv[m_b]
    stat, p = wilcoxon(piv[m_a], piv[m_b])
    win = float((d < 0).mean()) * 100
    print(f"  {m_a} vs {m_b}: n={len(piv):,} | "
          f"평균차 {d.mean():+.4f} | {m_a} 승률 {win:.1f}% | Wilcoxon p={p:.3g}")


def main() -> None:
    df = load()
    models = [m for m in fu.CMP_MODEL_ORDER if m in df.model.unique()]

    print("\n" + "=" * 74)
    print("[커버리지] 모델별 group_type='all' 평가 셀 수")
    print("=" * 74)
    cov = df[df.group_type == "all"].groupby("model").size()
    print(cov.reindex(models).to_string())

    print("\n" + "=" * 74)
    print("[Table 1] 전체 NMAE (온실 간 평균 ± SD)")
    print("=" * 74)
    t1 = table_overall(df, models)
    best = t1.NMAE.iloc[0]
    t1["vs_best_%"] = (t1.NMAE / best - 1) * 100
    print(t1.round(4).to_string(index=False))

    print("\n" + "=" * 74)
    print("[Table 2] 변수별 NMAE")
    print("=" * 74)
    print(table_by(df, models, "variable",
                   order=["Tin", "Tout", "RH", "CO2", "Rad"]).round(4).to_string())

    print("\n" + "=" * 74)
    print("[Table 3] 계절별 NMAE")
    print("=" * 74)
    print(table_by(df, models, "group_value", group_type="season",
                   order=["spring", "summer", "fall", "winter"]).round(4).to_string())

    print("\n" + "=" * 74)
    print("[Table 4] Gap 길이별 NMAE")
    print("=" * 74)
    print(table_by(df, models, "gap_length_h").round(4).to_string())

    print("\n" + "=" * 74)
    print("[Table 5] 시나리오별 NMAE")
    print("=" * 74)
    print(table_by(df, models, "scenario").round(4).to_string())

    print("\n" + "=" * 74)
    print("[짝지은 검정] 동일 평가 셀 기준")
    print("=" * 74)
    for a, b in [("TimesFM3.0", "TimesFM2.5"), ("TimesFM3.0", "CAFI"),
                 ("TimesFM3.0", "Chronos2"), ("CAFI", "TimesFM2.5")]:
        if a in df.model.unique() and b in df.model.unique():
            paired_test(df, a, b)

    out = _HERE / "../03_result/comparison_tfm3/summary_tables.csv"
    t1.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n요약 저장 → {out.resolve()}")


if __name__ == "__main__":
    main()

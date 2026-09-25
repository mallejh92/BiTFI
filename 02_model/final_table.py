"""
final_table.py
최종 비교표 — 각 모델을 context sweep으로 확정한 자기 최적 context에서 평가.

최적 context (03_result/context_sweep{,_tfm3}, 시나리오 A·gap 24/168h·10온실·3repeat):
  AutoGluon 계열/LI/SeasonalNaive : 720h  (학습 시 고정 / context 비민감)
  Chronos-2, TimesFM-2.5          : 1080h
  TimesFM-3.0 계열, CAFI 계열      : 1440h
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

_H = Path(__file__).resolve().parent
sys.path[:0] = [str(_H), str(_H/"figures")]
import fig_utils as fu

# (모델 → 최적 context 결과가 들어있는 디렉터리)
SRC = {
    "LI": "comparison", "SeasonalNaive": "comparison",
    "AG-LightGBM": "comparison", "AG-RandomForest": "comparison",
    "AG-DeepAR": "comparison", "AG-PatchTST": "comparison",
    "Chronos2": "comparison", "TimesFM2.5": "comparison", "CAFI": "comparison",
    "TimesFM3.0": "comparison_tfm3_ctx1440",
    "TimesFM3.0-MV": "comparison_tfm3_ctx1440",
    "TimesFM3.0-COV": "comparison_tfm3_ctx1440",
    "CAFI-TimesFM3": "comparison_cafi_tfm3",
    "Spatial-Ridge": "comparison_spatial",
    "TimesFM3.0-COV-SPA": "comparison_spatial",
    "CAFI-R1": "comparison_cafi_r1",
    "CAFI-TimesFM3-R1": "comparison_cafi_tfm3_r1",
    "DAFI-TimesFM3-fwd": "comparison_dafi_tfm3",
    "DAFI-TimesFM3": "comparison_dafi_tfm3",
    "DAFI-Chronos2": "comparison_dafi_chronos",
}
KIND = {"CAFI": "제안 (Chronos-2)", "CAFI-TimesFM3": "제안 (TimesFM-3.0)",
        "TimesFM3.0": "zero-shot FM", "TimesFM3.0-MV": "zero-shot FM",
        "TimesFM3.0-COV": "zero-shot FM", "TimesFM2.5": "zero-shot FM",
        "Chronos2": "zero-shot FM", "AG-LightGBM": "학습형",
        "AG-RandomForest": "학습형", "AG-DeepAR": "학습형", "AG-PatchTST": "학습형",
        "SeasonalNaive": "비학습", "LI": "비학습",
        "Spatial-Ridge": "공간축 통제", "TimesFM3.0-COV-SPA": "변수축+공간축",
        "CAFI-R1": "제안 (Chronos-2, 1라운드)", "CAFI-TimesFM3-R1": "제안 (TimesFM-3.0, 1라운드)",
        "DAFI-TimesFM3": "제안 DAFI (TimesFM-3.0)", "DAFI-Chronos2": "제안 DAFI (Chronos-2)",
        "DAFI-TimesFM3-fwd": "DAFI 전방전용 (ablation)"}
KEY = ["greenhouse","scenario","masked_vars","gap_length_h","repeat",
       "group_type","group_value","variable"]


def load() -> pd.DataFrame:
    frames = []
    for m, d in SRC.items():
        f = _H / f"../03_result/{d}/results.csv"
        if not f.exists():
            print(f"  ! {m}: {d}/results.csv 없음 (미실행) → skip"); continue
        sub = pd.read_csv(f)
        sub = sub[sub.model == m]
        if sub.empty:
            print(f"  ! {m}: {d}에 행 없음")
            continue
        frames.append(sub)
    df = pd.concat(frames, ignore_index=True)
    print("모델별 사용 context:")
    for m in SRC:
        s = df[df.model == m]
        if len(s):
            print(f"  {fu.CMP_MODEL_DISPLAY.get(m,m):22s} ctx={sorted(s.context_len.unique())} "
                  f"({len(s):,}행)")
    return df


def main() -> None:
    if os.environ.get("BITFI_CLEAN")!="0" and (_H/"../03_result/active_evaluation.json").exists():
        import build_prism_figures as b
        full,_=b.load_results();scores=b.site_scores(full[full.group_type=="all"])
        table=b.rowsummary(scores,full.model.unique()).sort_values("NMAE")
        out=_H/"../03_result/final_comparison.csv"
        table.to_csv(out,index=False,encoding="utf-8-sig")
        print("Clean-20260911: greenhouse-level NMAE and bootstrap intervals")
        print(table.to_string(index=False))
        print("Paired greenhouse tests (Figure 3; Holm adjusted):")
        print(pd.read_csv(b.DATA/"paired_greenhouse_tests.csv").to_string(index=False))
        return
    df = load()
    allr = df[df.group_type == "all"]
    D = fu.CMP_MODEL_DISPLAY

    def pg(m): return allr[allr.model==m].groupby("greenhouse").apply(fu.wavg_nmae, include_groups=False)
    def by(m, col, gt="all"):
        d = allr[allr.model==m] if gt=="all" else df[(df.group_type==gt)&(df.model==m)]
        c = "group_value" if gt!="all" else col
        return d.groupby(c).apply(fu.wavg_nmae, include_groups=False)

    rows=[]
    for m in SRC:
        if m not in set(df.model): continue
        s=pg(m); v=by(m,"variable"); sc=by(m,"scenario"); g=by(m,"gap_length_h"); se=by(m,None,"season")
        rows.append(dict(model=m, NMAE=s.mean(), SD=s.std(), ctx=int(df[df.model==m].context_len.iloc[0]),
            **{k:v.get(k,np.nan) for k in ["Tin","Tout","RH","CO2","Rad"]},
            A=sc.get("A"),B=sc.get("B"),C=sc.get("C"),
            g6=g.get(6),g24=g.get(24),g168=g.get(168),
            **{f"s_{k}":se.get(k,np.nan) for k in ["spring","summer","fall","winter"]}))
    T=pd.DataFrame(rows).set_index("model").sort_values("NMAE")

    top=T.index[0]
    def sig(m):
        if m==top: return "—"
        piv=allr[allr.model.isin([top,m])].pivot_table(index=KEY,columns="model",values="NMAE").dropna(subset=[top,m])
        _,p=wilcoxon(piv[top],piv[m])
        return "n.s." if p>=0.05 else ("p<0.001" if p<1e-3 else f"p={p:.3f}")

    def fmt(m,c,nd=4):
        x=T.loc[m,c]; b=T[c].min()
        return f"**{x:.{nd}f}**" if abs(x-b)<1e-9 else f"{x:.{nd}f}"

    print("\n### 최종 비교표 — 각 모델 최적 context (10 test 온실, NMAE ± 온실 간 SD)\n")
    print("| 순위 | 모델 | 유형 | ctx | NMAE ± SD | 1위 대비 | Tin | Tout | RH | CO2 | Rad |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for i,m in enumerate(T.index,1):
        nm=D.get(m,m)
        if m in ("CAFI","CAFI-TimesFM3"): nm=f"**{nm}**"
        print(f"| {i} | {nm} | {KIND[m]} | {T.loc[m,'ctx']}h | {fmt(m,'NMAE')} ± {T.loc[m,'SD']:.4f} | "
              f"{sig(m)} | {fmt(m,'Tin',3)} | {fmt(m,'Tout',3)} | {fmt(m,'RH',3)} | "
              f"{fmt(m,'CO2',3)} | {fmt(m,'Rad',3)} |")

    print("\n### 시나리오 · gap 길이 · 계절\n")
    print("| 순위 | 모델 | A | B | C | 6h | 24h | 168h | 봄 | 여름 | 가을 | 겨울 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i,m in enumerate(T.index,1):
        print(f"| {i} | {D.get(m,m)} | {fmt(m,'A')} | {fmt(m,'B')} | {fmt(m,'C')} | "
              f"{fmt(m,'g6')} | {fmt(m,'g24')} | {fmt(m,'g168')} | "
              f"{fmt(m,'s_spring')} | {fmt(m,'s_summer')} | {fmt(m,'s_fall')} | {fmt(m,'s_winter')} |")

    print("\n### 상위권 짝지은 검정 (동일 평가 셀)\n")
    TOP=list(T.index[:4])
    for i in range(len(TOP)):
        for j in range(i+1,len(TOP)):
            a,b=TOP[i],TOP[j]
            piv=allr[allr.model.isin([a,b])].pivot_table(index=KEY,columns="model",values="NMAE").dropna(subset=[a,b])
            d=piv[a]-piv[b]; _,p=wilcoxon(piv[a],piv[b])
            print(f"  {D.get(a,a):20s} vs {D.get(b,b):20s}: n={len(piv):,} "
                  f"평균차 {d.mean():+.5f}  {D.get(a,a)} 승률 {(d<0).mean()*100:4.1f}%  p={p:.3g}")

    print("\n### 메커니즘 기여도 (같은 백본·같은 context 1440h)\n")
    V={m:T.loc[m,'NMAE'] for m in T.index}
    for base,arms in [("TimesFM3.0",["TimesFM3.0-MV","TimesFM3.0-COV","CAFI-TimesFM3","CAFI-TimesFM3-R1",
                                     "TimesFM3.0-COV-SPA","DAFI-TimesFM3-fwd","DAFI-TimesFM3"])]:
        for a in arms:
            if a not in V: continue
            print(f"  {D[base]} {V[base]:.4f} → {D[a]:24s} {V[a]:.4f} : "
                  f"{(V[base]-V[a])/V[base]*100:5.1f}% 개선")
    print(f"  {D['Chronos2']} {V['Chronos2']:.4f} → {D['CAFI']:20s} {V['CAFI']:.4f} : "
          f"{(V['Chronos2']-V['CAFI'])/V['Chronos2']*100:5.1f}% 개선  (ctx 1080→1440)")

    out=_H/"../03_result/final_comparison.csv"
    T.to_csv(out, encoding="utf-8-sig")
    print(f"\n저장 → {out.resolve()}")


if __name__ == "__main__":
    main()

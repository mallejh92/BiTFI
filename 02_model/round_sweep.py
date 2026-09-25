"""
round_sweep.py
CAFI 반복 refinement의 라운드별 정확도 추적 (Supplementary Figure용).

목적: 시나리오 B·C에서 라운드가 진행될수록 gap 오차가 어떻게 변하는지를
      두 백본(Chronos-2, TimesFM-3.0)에서 동일 프로토콜로 기록한다.
      시나리오 A는 covariate가 전부 실측이라 R1에서 즉시 수렴하므로 제외.

프로토콜:
  test 온실 10곳 × 시나리오 {B, C} × gap {24, 168}h × repeat 3 (seed 고정)
  매 라운드(R0..R5) 출력의 gap 구간 MAE를 정답과 직접 비교 (정규화 단위)
  B: Tin/RH/CO2 동시 결측, C: 5변수 동시 결측

실행:
  python round_sweep.py chronos   # .venv
  python round_sweep.py tfm3      # .venv_tfm3
출력: ../03_result/round_sweep/rounds_{backbone}.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import gpu_utils  # noqa: F401
from preprocessing import preprocess_file
from models.foundation_model import CAFIImputation, CAFITimesFM3Imputation

VARS = ["Tin", "Tout", "RH", "CO2", "Rad"]
SCEN = {"B": ["Tin", "RH", "CO2"], "C": VARS}
GAPS = [24, 168]
N_REP = 3
SEED = 42
CTX = 1440
MAX_ROUNDS = 5


def main(backbone: str) -> None:
    split = json.load(open(_HERE / "../03_result/comparison/split.json"))
    out_dir = _HERE / "../03_result/round_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"rounds_{backbone}.csv"

    if backbone == "chronos":
        model = CAFIImputation(fine_tune=False, context_len=CTX, max_rounds=MAX_ROUNDS, name="sweep-chronos")
    else:
        model = CAFITimesFM3Imputation(context_len=CTX, max_rounds=MAX_ROUNDS, name="sweep-tfm3")

    # 라운드별 출력을 가로채는 훅 (알고리즘은 건드리지 않는다)
    captured: list[pd.DataFrame] = []
    _r0, _cr = model._round0_impute, model._covariate_round

    def r0(*a, **k):
        o = _r0(*a, **k); captured.append(o.copy()); return o

    def cr(*a, **k):
        o = _cr(*a, **k); captured.append(o.copy()); return o

    model._round0_impute, model._covariate_round = r0, cr

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    t_all = time.perf_counter()

    for fp in split["test"]:
        res = preprocess_file(Path(fp), include_covariates=False)
        if res is None:
            continue
        d = res["data"][[c for c in VARS if c in res["data"].columns]]
        if d.shape[1] < 5:
            continue
        gh = res["name"]; n = len(d)
        for scen, masked_vars in SCEN.items():
            for gl in GAPS:
                for rep in range(N_REP):
                    # gap 위치: 좌측에 CTX 확보(CAFI는 좌측 컨텍스트만 사용), 우측은 24h 여백,
                    # 정답이 완전한 구간만. 짧은 온실(PF_0025102_01, 2155h)도 포함되도록 한다.
                    hi = n - gl - 24
                    if hi <= CTX:
                        print(f"  [{gh}] {scen}/gap{gl}: 시계열 {n}h — CTX 확보 불가 → skip"); break
                    for _ in range(50):
                        gs = int(rng.integers(CTX, hi))
                        ge = gs + gl - 1
                        if np.isfinite(d.iloc[gs:ge + 1][masked_vars].values).all():
                            break
                    else:
                        continue
                    mask = pd.DataFrame(1.0, index=d.index, columns=d.columns)
                    mask.loc[mask.index[gs:ge + 1], masked_vars] = 0.0
                    masked = d.mask(mask == 0)
                    captured.clear()
                    t0 = time.perf_counter()
                    model.impute(masked, mask)
                    truth = d.iloc[gs:ge + 1][masked_vars].values
                    for r_idx, df in enumerate(captured):
                        pred = df.iloc[gs:ge + 1][masked_vars].values
                        per_var = np.mean(np.abs(pred - truth), axis=0)
                        for v, e in zip(masked_vars, per_var):
                            rows.append(dict(backbone=backbone, greenhouse=gh, scenario=scen,
                                             gap_length_h=gl, repeat=rep, round=r_idx,
                                             variable=v, mae_norm=float(e)))
                        rows.append(dict(backbone=backbone, greenhouse=gh, scenario=scen,
                                         gap_length_h=gl, repeat=rep, round=r_idx,
                                         variable="ALL", mae_norm=float(per_var.mean())))
                    print(f"  [{gh}] {scen}/gap{gl}/r{rep}: {len(captured)} rounds "
                          f"({time.perf_counter() - t0:.1f}s)")
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n완료 {len(df):,}행 → {out_csv}  ({(time.perf_counter() - t_all) / 60:.1f}분)")
    summ = (df[df.variable == "ALL"].groupby(["scenario", "round"])["mae_norm"].mean()
            .unstack("round"))
    print("\n[시나리오 × 라운드 평균 MAE]"); print(summ.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1])

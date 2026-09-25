"""
run_ablation.py
CAFI ablation 실험 — 어떤 요소가 성능에 기여하는지 격리한다.

CAFI(v1, Chronos-2) 구조:
  Round 0            = 순수 Chronos2(변수 독립 예측, covariate 없음)
  Round 1..max_rounds = 나머지 변수를 covariate로 쓰는 cross-variable refinement
  + 외부 covariate(WindSpeed/RadIn)를 입력 프레임에 포함하면 다른 변수 예측 시 활용

Ablation 축:
  CAFI-full     : rounds=5, 외부 covariate O   (기준 = 메인 비교의 CAFI)
  CAFI-noExtCov : rounds=5, 외부 covariate X   (WindSpeed/RadIn 기여도 격리 — "5변수만")
  CAFI-r1/r2/r3 : rounds=1/2/3, 외부 covariate O (cross-variable refinement 수렴 곡선)

"refinement 완전 제거(rounds=0)"는 Round 0만 남아 순수 Chronos2와 동일하므로,
메인 비교의 Chronos2 결과가 곧 r0 지점이다(여기서 rounds=0은 코드상 rnd 미정의라 제외).

메인 비교와 동일한 split.json / Test 온실 / 마스킹(Scenario·gap)을 재사용하고,
결과만 03_result/comparison/ablation.csv로 분리 저장한다(resume 지원).

실행:
  python run_ablation.py                         # 전체 variant × Test 전체
  python run_ablation.py --quick                 # 스모크: Test 2온실·gap 24h·1 repeat·Scenario A
  python run_ablation.py --variants CAFI-full,CAFI-noExtCov
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import gpu_utils  # noqa: F401  (import 시 GPU 설정 side-effect)
from preprocessing import preprocess_file
from data_split import load_split
from masking_v2 import create_gap_masks, SCENARIO_CONFIGS, RANDOM_SEED
from models.foundation_model import CAFIImputation

# run_comparison의 검증된 헬퍼를 그대로 재사용(중복 구현 금지).
from run_comparison import (
    _eval_one, _load_completed, _append_rows, make_group_labels,
    _stable_seed, TARGET_VARS,
)

# ──────────────────────────────────────────────────────────────────────────────
# Ablation variant 정의
# ──────────────────────────────────────────────────────────────────────────────

ABLATION_VARIANTS: dict[str, dict] = {
    "CAFI-full":     dict(max_rounds=5, ext_cov=True),
    "CAFI-noExtCov": dict(max_rounds=5, ext_cov=False),
    "CAFI-r1":       dict(max_rounds=1, ext_cov=True),
    "CAFI-r2":       dict(max_rounds=2, ext_cov=True),
    "CAFI-r3":       dict(max_rounds=3, ext_cov=True),
    # 외부 covariate를 제거한 최종 CAFI(=CAFI-noExtCov)의 refinement 수렴 곡선용.
    # 메인 CAFI가 외부 covariate를 쓰지 않으므로, 수렴 곡선의 중간 라운드도
    # ext_cov=False로 맞춰야 끝점(round 5)과 헤드라인 수치가 일치한다.
    "CAFI-noExt-r1": dict(max_rounds=1, ext_cov=False),
    "CAFI-noExt-r2": dict(max_rounds=2, ext_cov=False),
    "CAFI-noExt-r3": dict(max_rounds=3, ext_cov=False),
}


def build_variants(keys: list[str], context_len: int) -> dict[str, object]:
    """variant 이름 리스트 → CAFIImputation 인스턴스 dict.

    Chronos backbone은 model_id|device로 캐시 공유되어 여러 variant를 만들어도
    파이프라인 로드는 1회다. max_rounds/uses_ext_covariates만 인스턴스별로 다르다.
    """
    models: dict[str, object] = {}
    for name in keys:
        if name not in ABLATION_VARIANTS:
            raise ValueError(f"알 수 없는 variant: {name} (가능: {list(ABLATION_VARIANTS)})")
        cfg = ABLATION_VARIANTS[name]
        m = CAFIImputation(
            fine_tune=False, context_len=context_len,
            max_rounds=cfg["max_rounds"], name=name,
        )
        # _eval_one이 covariate 주입 여부를 판단하는 플래그(하드코딩된 이름 대신).
        m.uses_ext_covariates = cfg["ext_cov"]
        models[name] = m
        print(f"  [{name}] rounds={cfg['max_rounds']}, 외부covariate={'O' if cfg['ext_cov'] else 'X'}")
    return models


# ──────────────────────────────────────────────────────────────────────────────
# 메인 루프 (run_comparison의 Test 루프와 동일 구조, 모델만 variant)
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation(
    result_dir: Path,
    variant_keys: list[str],
    scenarios: list[str],
    gap_lengths_h: list[int],
    n_repeats: int,
    context_len: int,
    n_test: int | None,
    reset: bool = False,
) -> pd.DataFrame:
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "ablation.csv"
    if reset and csv_path.exists():
        csv_path.unlink()
        print(f"기존 ablation 결과 초기화: {csv_path}")

    # 메인 비교와 동일한 split 재사용(반드시 run_comparison이 먼저 split.json을 만들어야 함).
    split = load_split(result_dir)
    test_paths = [Path(p) for p in split["test"]]
    if n_test:
        test_paths = test_paths[:n_test]

    print("=" * 72)
    print("CAFI Ablation 실험")
    print(f"  variant: {variant_keys}")
    print(f"  Test 온실: {len(test_paths)} | 시나리오: {scenarios} | "
          f"Gap: {gap_lengths_h}h | repeat≤{n_repeats} | ctx={context_len}h")
    print("=" * 72)

    print("\n[variant 준비]")
    models = build_variants(variant_keys, context_len)

    completed = _load_completed(csv_path)
    if completed:
        print(f"재개 모드: {len(completed)}개 (온실×variant×시나리오×gap×repeat) 완료됨")

    for fp in test_paths:
        res = preprocess_file(fp, include_covariates=True)
        if res is None:
            continue
        data = res["data"]
        scalers = res["scaler"]
        greenhouse = res["name"]
        target_cols = [c for c in TARGET_VARS if c in data.columns]
        if len(target_cols) < len(TARGET_VARS):
            print(f"  {greenhouse}: 타깃 변수 부족 → skip")
            continue
        covariate_cols = [c for c in data.columns if c not in TARGET_VARS]

        # NMAE 정규화 상수(온실별 변수 전체 관측 range) — run_comparison과 동일 규약.
        data_raw = res["data_raw"]
        norm_ranges = {
            c: float(data_raw[c].max() - data_raw[c].min())
            for c in target_cols if c in data_raw.columns
        }

        test_norm = data[target_cols]
        full_norm = data[target_cols + covariate_cols]
        full_valid = pd.DataFrame(
            np.where(full_norm.isna().values, 0.0, 1.0).astype(np.float32),
            index=full_norm.index, columns=full_norm.columns,
        )
        group_labels = make_group_labels(test_norm.index)

        print(f"\n{'='*72}\n[{greenhouse}] Test {len(test_norm)}h "
              f"| 계절: {sorted(set(group_labels['season']))} | covariate: {covariate_cols}")

        for scenario in scenarios:
            for masked_vars in SCENARIO_CONFIGS.get(scenario, []):
                for gap_h in gap_lengths_h:
                    masks = create_gap_masks(
                        test_data=full_norm,
                        original_valid_mask=full_valid,
                        scenario=scenario, gap_length_h=gap_h,
                        masked_vars=masked_vars, context_len_h=context_len,
                        n_repeats=n_repeats,
                        random_seed=RANDOM_SEED + _stable_seed(f"{greenhouse}{scenario}{gap_h}"),
                    )
                    if not masks:
                        continue
                    for mr in masks:
                        for vname, model in models.items():
                            key = (greenhouse, vname, scenario, ",".join(mr.masked_vars),
                                   gap_h, mr.repeat)
                            if key in completed:
                                continue
                            t0 = time.perf_counter()
                            try:
                                rows = _eval_one(
                                    vname, model, mr, scalers, group_labels,
                                    greenhouse, covariate_cols, target_cols, context_len,
                                    base=None, norm_ranges=norm_ranges,
                                )
                            except Exception as e:
                                print(f"    [{vname}] 실패: {e}")
                                traceback.print_exc()
                                continue
                            _append_rows(csv_path, rows)
                            completed.add(key)
                            print(f"    [{vname}] {scenario}/{','.join(mr.masked_vars)}"
                                  f"/gap{gap_h}/r{mr.repeat} ({time.perf_counter()-t0:.1f}s)")

    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    print(f"\n완료: {len(df)}행 → {csv_path}")
    return df


def main():
    p = argparse.ArgumentParser(description="CAFI ablation 실험")
    p.add_argument("--result-dir", type=str,
                   default=str(Path(__file__).resolve().parent.parent / "03_result" / "comparison"))
    p.add_argument("--variants", type=str, default=",".join(ABLATION_VARIANTS.keys()),
                   help=f"쉼표구분. 가능: {','.join(ABLATION_VARIANTS.keys())}")
    p.add_argument("--scenarios", type=str, default="A,B,C")
    p.add_argument("--gap-lengths", type=str, default="6,12,24,72,168")
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--context-len", type=int, default=720)
    p.add_argument("--n-test", type=int, default=None, help="Test 온실 수 제한(기본 전체)")
    p.add_argument("--reset", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="스모크: Test 2온실·Scenario A·gap 24h·1 repeat")
    args = p.parse_args()

    if args.quick:
        variants = ["CAFI-full", "CAFI-noExtCov"]
        scenarios = ["A"]
        gaps = [24]
        n_repeats = 1
        n_test = 2
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
        scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        gaps = [int(g) for g in args.gap_lengths.split(",") if g.strip()]
        n_repeats = args.n_repeats
        n_test = args.n_test

    run_ablation(
        result_dir=Path(args.result_dir),
        variant_keys=variants,
        scenarios=scenarios,
        gap_lengths_h=gaps,
        n_repeats=n_repeats,
        context_len=args.context_len,
        n_test=n_test,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()

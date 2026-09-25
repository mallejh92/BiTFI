"""
run_experiment_v2.py
논문용 온실 환경 시계열 결측 복원 실험 (v2)

실험 설계:
  데이터:     01_data/PF_*.xlsx (7개 온실)
  변수:       Tin, Tout, RH, CO2, Rad (5개)
  시나리오:   A (개별), B (부분 다변량), C (전체 다변량)
  Gap length: 6, 12, 24, 72, 168 h
  반복:       최대 10회 (비중첩 블록)
  Context:    720 h (30일) 고정

비교 모델:
  LinearInterp    — 단순 선형 보간 (baseline)
  SeasonalNaive   — 계절 나이브 (season=24h, context=720h)
  PatchTST        — Patch Transformer (온실 데이터로 학습)
  Chronos2        — Chronos-2 zero-shot foundation model
  CAFI            — Chronos-2 + covariate-aware iterative refinement (제안 방법)

결과:
  03_result/experiment_v2/results.csv   — 전체 결과
  컬럼: greenhouse_id, model, scenario, masked_vars, gap_length_h,
         context_length_h, variable, repeat, MAE, RMSE, nRMSE, R2, n_eval

실행:
  python run_experiment_v2.py [--greenhouses PF_0020209_01,...] \\
                               [--scenarios A,B,C] \\
                               [--gap-lengths 6,12,24,72,168] \\
                               [--n-repeats 10] \\
                               [--context-len 720] \\
                               [--models linear,seasonalnaive,patchtst,chronos2,cafi] \\
                               [--data-dir ../01_data] \\
                               [--result-dir ../03_result/experiment_v2] \\
                               [--reset]
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

# legacy/ 로 이동했으므로 02_model/(부모)까지 sys.path에 포함해야
# preprocessing·evaluate·masking·gpu_utils·models 패키지를 찾을 수 있다.
_SCRIPT_DIR = Path(__file__).resolve().parent
_THIS_DIR = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# GPU 호환성: torch import 이전에 반드시 실행
import gpu_utils  # noqa: F401

import numpy as np
import pandas as pd

from preprocessing import (
    preprocess_file,
    train_test_split,
    fit_scalers_from_split,
    TARGET_COLS,
)
from masking_v2 import (
    create_gap_masks,
    SCENARIO_CONFIGS,
    GAP_LENGTHS_H,
    CONTEXT_LEN_H,
    N_REPEATS,
    RANDOM_SEED,
    MaskResult,
)
from evaluate import compute_metrics_v2

from models.linear_interpolation import LinearInterpolation
from models.seasonal_naive import SeasonalNaiveImputation
from models.foundation_model import CAFIImputation, ChronosImputation

TARGET_VARS = list(TARGET_COLS.values())  # ['Tin', 'Tout', 'RH', 'CO2', 'Rad']

# ── CSV 컬럼 순서 ─────────────────────────────────────────────────────────────
_CSV_COLS = [
    "greenhouse_id", "model", "scenario", "masked_vars",
    "gap_length_h", "context_length_h",
    "variable", "repeat",
    "MAE", "RMSE", "nRMSE", "R2", "n_eval",
    "impute_seconds",
]


# ──────────────────────────────────────────────────────────────────────────────
# 모델 빌더
# ──────────────────────────────────────────────────────────────────────────────

def _build_models(
    model_keys: list[str],
    context_len: int,
    result_dir: Path | None = None,
) -> dict[str, object]:
    """
    모델 키 목록 → 모델 인스턴스 dict.
    모델은 온실당 1회 fit 후 모든 시나리오·갭에서 재사용한다.
    """
    models: dict[str, object] = {}
    for key in model_keys:
        k = key.strip().lower()

        if k == "linear":
            models["LinearInterp"] = LinearInterpolation()

        elif k == "seasonalnaive":
            models["SeasonalNaive"] = SeasonalNaiveImputation(
                season_length=24,
                context_len=context_len,
            )

        elif k == "patchtst":
            try:
                from models.patchtst_model import PatchTSTImputation
                models["PatchTST"] = PatchTSTImputation(
                    context_window=context_len,
                    patch_len=24,        # 24h = 일주기 정렬
                    d_model=128,
                    d_ff=256,
                    mask_type="mixed",   # 랜덤 + 블록 혼합 학습
                    epochs=100,
                    patience=15,
                )
            except Exception as e:
                warnings.warn(f"PatchTST 로드 실패: {e}")
                raise  # 경고만 하지 말고 오류를 노출해 원인 파악 가능하게

        elif k in ("recursivetabular", "ml"):
            try:
                from models.autogluon_model import AutoGluonImputation
                ag_save = (result_dir / "models" / "RecursiveTabular") if result_dir is not None else None
                models["RecursiveTabular"] = AutoGluonImputation(
                    context_len=context_len,
                    time_limit=120,
                    hyperparameters={"RecursiveTabular": {}},
                    name="RecursiveTabular",
                    save_path=ag_save,
                )
            except Exception as e:
                warnings.warn(f"RecursiveTabular 로드 실패: {e}")
                raise

        elif k == "chronos2":
            models["Chronos2"] = ChronosImputation(
                context_len=context_len,
                fine_tune=False,
            )

        elif k == "cafi":
            models["CAFI"] = CAFIImputation(
                context_len=context_len,
                max_rounds=5,
                tol=1e-3,
                use_std_normalization=True,
                use_time_covariates=True,
            )

        else:
            warnings.warn(f"알 수 없는 모델 키 → 건너뜀: {key}")

    return models


# ──────────────────────────────────────────────────────────────────────────────
# 중간 저장 / 재개
# ──────────────────────────────────────────────────────────────────────────────

def _load_completed(csv_path: Path) -> set[tuple]:
    """이미 완료된 (greenhouse_id, model, scenario, masked_vars, gap_length_h, variable, repeat) 집합."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(zip(
            df["greenhouse_id"], df["model"], df["scenario"],
            df["masked_vars"], df["gap_length_h"], df["variable"], df["repeat"],
        ))
    except Exception:
        return set()


def _append_rows(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df[[c for c in _CSV_COLS if c in df.columns]]
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False, encoding="utf-8-sig")


# ──────────────────────────────────────────────────────────────────────────────
# 단일 (온실 × 시나리오 × 변수조합 × gap × 반복 × 모델) 평가
# ──────────────────────────────────────────────────────────────────────────────

def _run_one_mask(
    mask_result: MaskResult,
    models: dict[str, object],
    scalers: dict,
    greenhouse_id: str,
    completed: set[tuple],
    csv_path: Path,
) -> list[dict]:
    """하나의 MaskResult에 대해 모든 모델을 실행하고 결과를 저장한다."""
    vars_key = ",".join(mask_result.masked_vars)
    new_rows: list[dict] = []

    for model_name, model in models.items():
        # ── resume: 이미 완료된 조합 건너뜀 ──
        # 변수별로 체크 (eval_mask가 True인 변수만 실제 평가 대상)
        vars_to_eval = [
            v for v in mask_result.masked_vars
            if v in mask_result.eval_mask.columns and mask_result.eval_mask[v].any()
        ]
        already_done = all(
            (greenhouse_id, model_name, mask_result.scenario, vars_key,
             mask_result.gap_length_h, v, mask_result.repeat) in completed
            for v in vars_to_eval
        )
        if already_done and vars_to_eval:
            continue

        # ── imputation ──
        t0 = time.perf_counter()
        try:
            imputed = model.impute(
                mask_result.masked_data,
                mask_result.effective_mask,
            )
        except Exception as e:
            print(f"    [{model_name}] impute 실패: {e}")
            traceback.print_exc()
            continue
        impute_secs = time.perf_counter() - t0

        # ── 평가 ──
        try:
            metrics = compute_metrics_v2(
                ground_truth=mask_result.ground_truth,
                imputed=imputed,
                eval_mask=mask_result.eval_mask,
                scalers=scalers,
            )
        except Exception as e:
            print(f"    [{model_name}] compute_metrics 실패: {e}")
            continue

        # ── 결과 행 생성 (평가 가능한 변수만) ──
        rows: list[dict] = []
        for var, m in metrics.items():
            if m["n_eval"] == 0:
                continue
            rows.append({
                "greenhouse_id":  greenhouse_id,
                "model":          model_name,
                "scenario":       mask_result.scenario,
                "masked_vars":    vars_key,
                "gap_length_h":   mask_result.gap_length_h,
                "context_length_h": mask_result.context_len_h,
                "variable":       var,
                "repeat":         mask_result.repeat,
                "MAE":            round(m["MAE"],   6) if not np.isnan(m["MAE"])   else np.nan,
                "RMSE":           round(m["RMSE"],  6) if not np.isnan(m["RMSE"])  else np.nan,
                "nRMSE":          round(m["nRMSE"], 6) if not np.isnan(m["nRMSE"]) else np.nan,
                "R2":             round(m["R2"],    6) if not np.isnan(m["R2"])    else np.nan,
                "n_eval":         m["n_eval"],
                "impute_seconds": round(impute_secs, 3),
            })

        if rows:
            _append_rows(csv_path, rows)
            for r in rows:
                completed.add((
                    r["greenhouse_id"], r["model"], r["scenario"],
                    r["masked_vars"], r["gap_length_h"], r["variable"], r["repeat"],
                ))
            new_rows.extend(rows)

        mae_vals = [r["MAE"] for r in rows if not (isinstance(r["MAE"], float) and np.isnan(r["MAE"]))]
        mae_str = f"{np.mean(mae_vals):.4f}" if mae_vals else "n/a"
        print(
            f"    [{model_name}] gap={mask_result.gap_length_h}h "
            f"repeat={mask_result.repeat} "
            f"MAE={mae_str} ({impute_secs:.1f}s)"
        )

    return new_rows


# ──────────────────────────────────────────────────────────────────────────────
# 메인 실험 루프
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(
    model_keys: list[str],
    scenarios: list[str],
    gap_lengths_h: list[int],
    n_repeats: int,
    context_len: int,
    data_dir: Path,
    result_dir: Path,
    filter_greenhouses: list[str] | None = None,
    reset: bool = False,
) -> pd.DataFrame:
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "results.csv"

    if reset and csv_path.exists():
        csv_path.unlink()
        print(f"기존 결과 초기화: {csv_path}")

    completed = _load_completed(csv_path)
    if completed:
        print(f"재개 모드: {len(completed)}개 조합 이미 완료")

    data_files = sorted(data_dir.glob("PF_*.xlsx"))
    if filter_greenhouses:
        data_files = [f for f in data_files if f.stem in filter_greenhouses]

    print("=" * 72)
    print("온실 환경 시계열 결측 복원 실험 v2")
    print(f"  데이터: {data_dir} ({len(data_files)}개)")
    print(f"  모델:   {model_keys}")
    print(f"  시나리오: {scenarios}")
    print(f"  Gap:    {gap_lengths_h} h")
    print(f"  반복:   최대 {n_repeats}회  |  Context: {context_len}h")
    print(f"  결과:   {csv_path}")
    print("=" * 72)

    for fp in data_files:
        result = preprocess_file(fp)
        if result is None:
            continue
        if not all(v in result["data"].columns for v in TARGET_VARS):
            print(f"  {fp.name}: 타깃 변수 부족 → skip")
            continue

        greenhouse_id = result["name"]
        print(f"\n{'='*72}")
        print(f"[{greenhouse_id}]")
        print("=" * 72)

        # ── train/test split (80/20, 시간 순) ──
        train, test = train_test_split(result, test_ratio=0.2)

        # ── scaler: train data만으로 fit ──
        scalers, train_norm, test_norm = fit_scalers_from_split(
            train["data_raw"], test["data_raw"]
        )

        # ── original_valid_mask: 원본 결측 위치 (1=관측, 0=원본결측) ──
        original_valid_mask = pd.DataFrame(
            np.where(test_norm.isna().values, 0.0, 1.0).astype(np.float32),
            index=test_norm.index,
            columns=test_norm.columns,
        )

        print(f"  Train: {len(train_norm)}h  |  Test: {len(test_norm)}h")
        miss_rate = 1.0 - original_valid_mask.mean()
        print("  Test 원본 결측률:")
        for v, r in miss_rate.items():
            if r > 0:
                print(f"    {v}: {r*100:.1f}%")

        # ── 모델 인스턴스 생성 및 학습 (온실당 1회) ──
        print("\n  모델 학습 중...")
        models = _build_models(model_keys, context_len, result_dir)
        for mname, model in models.items():
            try:
                t0 = time.perf_counter()
                model.fit(train_norm)
                print(f"    [{mname}] fit 완료 ({time.perf_counter()-t0:.1f}s)")
            except Exception as e:
                warnings.warn(f"  [{mname}] fit 실패: {e}")

        # ── 시나리오 루프 ──
        for scenario in scenarios:
            var_combos = SCENARIO_CONFIGS.get(scenario, [])
            print(f"\n  --- Scenario {scenario} ---")

            for masked_vars in var_combos:
                vars_key = ",".join(masked_vars)
                print(f"\n  [{scenario}] masked_vars={vars_key}")

                for gap_h in gap_lengths_h:
                    masks = create_gap_masks(
                        test_data=test_norm,
                        original_valid_mask=original_valid_mask,
                        scenario=scenario,
                        gap_length_h=gap_h,
                        masked_vars=masked_vars,
                        context_len_h=context_len,
                        n_repeats=n_repeats,
                        random_seed=RANDOM_SEED + hash(f"{greenhouse_id}{scenario}{gap_h}{vars_key}") % 9999,
                    )

                    if not masks:
                        print(f"    gap={gap_h}h: 유효한 갭 없음 → skip")
                        continue

                    print(f"    gap={gap_h}h: {len(masks)}개 반복")

                    for mask_result in masks:
                        _run_one_mask(
                            mask_result=mask_result,
                            models=models,
                            scalers=scalers,
                            greenhouse_id=greenhouse_id,
                            completed=completed,
                            csv_path=csv_path,
                        )

    # ── 최종 결과 출력 ──
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\n\n{'='*72}")
        print(f"실험 완료. 총 {len(df)}행 → {csv_path}")
        _print_summary(df)
        return df

    print("\n수집된 결과 없음.")
    return pd.DataFrame()


# ──────────────────────────────────────────────────────────────────────────────
# 요약 출력
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame) -> None:
    print("\n[모델 × 시나리오 평균 RMSE]")
    try:
        tbl = (
            df.groupby(["model", "scenario"])["RMSE"]
            .mean()
            .round(4)
            .unstack("scenario")
        )
        print(tbl.to_string())
    except Exception:
        pass

    print("\n[모델 × Gap length 평균 RMSE]")
    try:
        tbl2 = (
            df.groupby(["model", "gap_length_h"])["RMSE"]
            .mean()
            .round(4)
            .unstack("gap_length_h")
        )
        print(tbl2.to_string())
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="온실 환경 시계열 결측 복원 실험 v2")
    p.add_argument("--greenhouses", type=str, default="",
                   help="실험할 온실 ID (쉼표 구분, 기본=전체)")
    p.add_argument("--scenarios", type=str, default="A,B,C",
                   help="시나리오 (A,B,C)")
    p.add_argument("--gap-lengths", type=str, default="6,12,24,72,168",
                   help="Gap length 목록 (h)")
    p.add_argument("--n-repeats", type=int, default=N_REPEATS,
                   help="최대 반복 수")
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN_H,
                   help="Context length (h)")
    p.add_argument("--models", type=str,
                   default="linear,seasonalnaive,patchtst,chronos2,cafi",
                   help="비교 모델 (linear,seasonalnaive,patchtst,chronos2,cafi)")
    p.add_argument("--data-dir", type=str, default="../01_data")
    p.add_argument("--result-dir", type=str, default="../03_result/experiment_v2")
    p.add_argument("--reset", action="store_true", default=False,
                   help="기존 results.csv 삭제 후 처음부터 실행")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    model_keys   = [m.strip() for m in args.models.split(",")      if m.strip()]
    scenarios    = [s.strip() for s in args.scenarios.split(",")   if s.strip()]
    gap_lengths  = [int(g)    for g in args.gap_lengths.split(",") if g.strip()]
    greenhouses  = [g.strip() for g in args.greenhouses.split(",") if g.strip()] or None

    data_dir   = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    if not data_dir.is_absolute():
        data_dir = (_THIS_DIR / data_dir).resolve()
    if not result_dir.is_absolute():
        result_dir = (_THIS_DIR / result_dir).resolve()

    run_experiment(
        model_keys=model_keys,
        scenarios=scenarios,
        gap_lengths_h=gap_lengths,
        n_repeats=args.n_repeats,
        context_len=args.context_len,
        data_dir=data_dir,
        result_dir=result_dir,
        filter_greenhouses=greenhouses,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()

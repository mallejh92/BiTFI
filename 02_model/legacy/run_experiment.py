"""
run_experiment.py
온실 시계열 보간 실험 메인 러너

실험 설계:
  - 대상: 01_data/PF_*.xlsx (7개 온실)
  - 마스킹: block masking, loss_rate ∈ [0.1, 0.3, 0.5, 0.7, 0.9]
  - 모델: AutoGluon, CAFI (CAFIv2), LinearInterpolation
  - 평가: MSE, MAE, NMAE (역정규화 후)
  - 결과: 03_result/autogluon/results_all.csv (loss_rate마다 누적 저장)

설계 원칙:
  - AutoGluon은 온실당 1회만 학습 (predictor 재사용)
  - 이미 저장된 predictor가 있으면 재학습 없이 로드
  - (greenhouse, loss_rate, model) 단위로 결과를 즉시 저장 → 중단 후 재개 가능

실행:
  python run_experiment.py [--models MODEL1,MODEL2] [--loss-rates 0.1,0.3,0.5]
                           [--context-len 336] [--data-dir ../01_data]
                           [--result-dir ../03_result/autogluon]
                           [--time-limit 600]
    python run_experiment.py --reset --expand-submodels
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
from pathlib import Path

# GPU 호환성 자동 선택 (Blackwell SM≥10 → A6000으로 폴백)
# 반드시 torch import 이전에 위치해야 한다.
# legacy/ 로 이동했으므로 02_model/(부모)까지 sys.path에 포함해야
# preprocessing·evaluate·masking·gpu_utils·models 패키지를 찾을 수 있다.
_SCRIPT_DIR = Path(__file__).resolve().parent
_THIS_DIR = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import gpu_utils  # noqa: F401, E402

import numpy as np
import pandas as pd

from preprocessing import (  # noqa: E402
    preprocess_file,
    train_test_split,
    fit_scalers_from_split,
    TARGET_COLS,
)
from masking import apply_masking  # noqa: E402
from evaluate import print_results_table, compute_metrics  # noqa: E402

from models.linear_interpolation import LinearInterpolation  # noqa: E402
from models.autogluon_model import AutoGluonImputation  # noqa: E402
from models.foundation_model import CAFIv2Imputation  # noqa: E402

TARGET_VARS = list(TARGET_COLS.values())  # ['Tin','Tout','RH','CO2','Rad']

MASK_TYPE = "block"
BLOCK_HOURS = 48
RANDOM_SEED = 42


# ──────────────────────────────────────────────
# 모델 메타정보 (이름 → model_key, backbone)
# ──────────────────────────────────────────────

_MODEL_META: dict[str, tuple[str, str]] = {
    "autogluon": ("AutoGluonImputation", "AutoGluon-best_quality"),
    "cafi":      ("CAFIv2Imputation",    "TimesFM+Ridge"),
    "linear":    ("LinearInterpolation", "LinearInterp"),
}


# ──────────────────────────────────────────────
# 온실당 1회 모델 학습 (AutoGluon 재사용 핵심)
# ──────────────────────────────────────────────

def _train_models_for_greenhouse(
    model_names: list[str],
    context_len: int,
    time_limit: int,
    train_df: pd.DataFrame,
    model_save_dir: Path | None,
    greenhouse: str,
) -> dict[str, tuple]:
    """
    온실 하나에 대해 모든 모델을 학습(또는 로드)한다.

    Returns
    -------
    dict: model_key → (model_instance, model_key, backbone, fit_seconds)
    """
    trained: dict[str, tuple] = {}

    for name in model_names:
        key = name.strip().lower()
        mk, bb = _MODEL_META.get(key, (name, name))

        print(f"\n  [{greenhouse}] 모델 준비: {mk}")

        if key == "autogluon":
            ag_save = (model_save_dir / greenhouse) if model_save_dir else None
            # 이미 저장된 predictor가 있으면 재학습 없이 로드
            if ag_save and (ag_save / "predictor").exists():
                try:
                    m = AutoGluonImputation.load(
                        ag_save,
                        context_len=context_len,
                        time_limit=time_limit,
                    )
                    print(f"    → 기존 predictor 로드 완료 (학습 생략): {ag_save}")
                    trained[key] = (m, mk, bb, 0.0)
                    continue
                except Exception as e:
                    warnings.warn(f"    predictor 로드 실패 ({e}), 재학습합니다.")

            m = AutoGluonImputation(
                context_len=context_len,
                time_limit=time_limit,
                save_path=ag_save,
            )
            t0 = time.perf_counter()
            m.fit(train_df)
            fit_secs = time.perf_counter() - t0

        elif key == "cafi":
            m = CAFIv2Imputation(context_len=context_len)
            t0 = time.perf_counter()
            m.fit(train_df)
            fit_secs = time.perf_counter() - t0

        elif key == "linear":
            m = LinearInterpolation()
            t0 = time.perf_counter()
            m.fit(train_df)
            fit_secs = time.perf_counter() - t0

        else:
            raise ValueError(f"알 수 없는 모델 키: {name}")

        trained[key] = (m, mk, bb, fit_secs)

    return trained


# ──────────────────────────────────────────────
# 단일 (모델 × loss_rate) 보간·평가
# ──────────────────────────────────────────────

def _impute_and_eval(
    model,
    model_key: str,
    backbone: str,
    fit_seconds: float,
    test_masked: pd.DataFrame,
    mask_matrix: pd.DataFrame,
    ground_truth: pd.DataFrame,
    original_mask: pd.DataFrame,
    scalers: dict,
    greenhouse: str,
    loss_rate: float,
    context_len: int,
) -> list[dict]:
    display_name = getattr(model, "name", model_key)

    t1 = time.perf_counter()
    imputed = model.impute(test_masked, mask_matrix)
    impute_seconds = time.perf_counter() - t1
    total_seconds = fit_seconds + impute_seconds

    metrics = compute_metrics(
        ground_truth=ground_truth,
        imputed=imputed,
        original_mask=original_mask,
        artificial_mask=mask_matrix,
        scalers=scalers,
    )

    rows: list[dict] = []
    for var, m in metrics.items():
        rows.append({
            "model":          display_name,
            "greenhouse":     greenhouse,
            "variable":       var,
            "mask_type":      MASK_TYPE,
            "loss_rate":      loss_rate,
            "MSE":            m["MSE"],
            "MAE":            m["MAE"],
            "NMAE":           m["NMAE"],
            "n_eval":         m["n_eval"],
            "context_len":    context_len,
            "fit_seconds":    round(fit_seconds, 3),
            "impute_seconds": round(impute_seconds, 3),
            "total_seconds":  round(total_seconds, 3),
            "model_key":      model_key,
            "backbone":       backbone,
        })
    return rows


# ──────────────────────────────────────────────
# 중간 저장 / 재개 헬퍼
# ──────────────────────────────────────────────

_CSV_COLS = [
    "model", "greenhouse", "variable", "mask_type", "loss_rate",
    "MSE", "MAE", "NMAE", "n_eval",
    "context_len", "fit_seconds", "impute_seconds", "total_seconds",
    "model_key", "backbone",
]


def _load_completed(csv_path: Path) -> set[tuple]:
    """이미 완료된 (greenhouse, loss_rate, model) 조합 집합을 반환한다."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(zip(df["greenhouse"], df["loss_rate"], df["model"]))
    except Exception:
        return set()


def _append_rows(csv_path: Path, rows: list[dict]) -> None:
    """결과 행을 CSV에 즉시 추가한다 (헤더는 최초 1회만)."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    df = df[[c for c in _CSV_COLS if c in df.columns]]
    header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=header, index=False, encoding="utf-8-sig")


# ──────────────────────────────────────────────
# 메인 실험 루프
# ──────────────────────────────────────────────

def run_experiment(
    models: list[str],
    loss_rates: list[float],
    context_len: int,
    data_dir: Path,
    result_dir: Path,
    time_limit: int,
    save_models: bool = True,
    expand_submodels: bool = False,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    model_save_dir = (result_dir / "models") if save_models else None
    if model_save_dir is not None:
        model_save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = result_dir / "results_all.csv"
    completed = _load_completed(csv_path)
    if completed:
        print(f"  재개 모드: 이미 완료된 조합 {len(completed)}개 건너뜀")

    data_files = sorted(data_dir.glob("PF_*.xlsx"))
    if not data_files:
        print(f"데이터 파일 없음: {data_dir}/PF_*.xlsx")
        return pd.DataFrame()

    print("=" * 70)
    print("온실 시계열 보간 실험")
    print(f"  데이터:    {data_dir} ({len(data_files)}개 파일)")
    print(f"  모델:      {models}")
    print(f"  loss_rate: {loss_rates}")
    print(f"  context:   {context_len}")
    print(f"  마스킹:    {MASK_TYPE} (block_hours={BLOCK_HOURS})")
    print(f"  결과 저장: {csv_path} (loss_rate마다 즉시 저장)")
    print(f"  sub-model 개별 결과: {'ON' if expand_submodels else 'OFF (--expand-submodels로 활성화)'}")
    print("=" * 70)

    for fp in data_files:
        result = preprocess_file(fp)
        if result is None:
            print(f"  {fp.name}: 전처리 결과 없음 → skip")
            continue
        if not all(v in result["data"].columns for v in TARGET_VARS):
            print(f"  {fp.name}: 타깃 변수 부족 → skip")
            continue

        greenhouse = result["name"]
        freq = result.get("freq", "1H")

        # 이 온실이 완료된 (loss_rate, model) 쌍을 직접 세어서 판단
        done_for_gh = {(lr, mn) for (gh, lr, mn) in completed if gh == greenhouse}
        if len(done_for_gh) >= len(loss_rates) * len(models):
            print(f"\n[{greenhouse}] 모든 조합 완료됨 → 건너뜀")
            continue

        print(f"\n{'=' * 70}")
        print(f"[{greenhouse}]")
        print("=" * 70)

        # ── train/test split (시간순 80/20) ──
        train, test = train_test_split(result, test_ratio=0.2)
        scalers, train_norm, test_norm = fit_scalers_from_split(
            train["data_raw"], test["data_raw"]
        )
        original_mask = pd.DataFrame(
            np.where(test_norm.isna().values, 0.0, 1.0).astype(np.float32),
            index=test_norm.index,
            columns=test_norm.columns,
        )

        # ── 온실당 1회 모델 학습 / predictor 로드 ──
        try:
            trained = _train_models_for_greenhouse(
                model_names=models,
                context_len=context_len,
                time_limit=time_limit,
                train_df=train_norm,
                model_save_dir=model_save_dir,
                greenhouse=greenhouse,
            )
        except Exception as e:
            print(f"  [ERROR] {greenhouse} 모델 준비 실패: {e}")
            traceback.print_exc()
            continue

        # ── AutoGluon sub-model 확장 (--expand-submodels 시) ──
        if expand_submodels and "autogluon" in trained:
            ag_main, ag_mk, ag_bb, ag_fsecs = trained["autogluon"]
            for wrapper in ag_main.create_submodel_wrappers():
                wkey = f"autogluon_{wrapper.sub_model_name}"
                trained[wkey] = (wrapper, "AutoGluonImputation", wrapper.sub_model_name, 0.0)

        # ── loss_rate 루프 ──
        for loss_rate in loss_rates:
            print(f"\n{'-' * 70}")
            print(f"[{greenhouse}] loss_rate={loss_rate:.2f}")
            print("-" * 70)

            masked_data, mask_matrix, ground_truth = apply_masking(
                test_norm,
                mask_type=MASK_TYPE,
                loss_rate=loss_rate,
                freq=freq,
                block_hours=BLOCK_HOURS,
                random_seed=RANDOM_SEED,
            )

            lr_rows: list[dict] = []

            for model_name in models:
                key = model_name.strip().lower()
                if key not in trained:
                    print(f"  [{model_name}] 준비되지 않음 → skip")
                    continue

                model, mk, bb, fit_secs = trained[key]
                display_name = getattr(model, "name", mk)

                # 이미 완료된 조합은 건너뜀
                if (greenhouse, loss_rate, display_name) in completed:
                    print(f"  [{display_name}] 이미 완료됨 → 건너뜀")
                    continue

                print(f"  [{display_name}] 보간 중... (fit={fit_secs:.1f}s)")
                try:
                    rows = _impute_and_eval(
                        model=model,
                        model_key=mk,
                        backbone=bb,
                        fit_seconds=fit_secs,
                        test_masked=masked_data,
                        mask_matrix=mask_matrix,
                        ground_truth=ground_truth,
                        original_mask=original_mask,
                        scalers=scalers,
                        greenhouse=greenhouse,
                        loss_rate=loss_rate,
                        context_len=context_len,
                    )
                    lr_rows.extend(rows)
                    # 즉시 CSV에 저장 (충돌 복구 가능)
                    _append_rows(csv_path, rows)
                    completed.add((greenhouse, loss_rate, display_name))

                    mae_vals = [r["MAE"] for r in rows if not np.isnan(r.get("MAE", np.nan))]
                    if mae_vals:
                        print(f"    → 완료. 평균 MAE={np.mean(mae_vals):.4f}")

                except Exception as e:
                    print(f"  [ERROR] {display_name} @ {greenhouse} lr={loss_rate}: {e}")
                    traceback.print_exc()

    # ── 최종 결과 읽기 + 출력 ──
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df.sort_values(["model", "mask_type", "loss_rate", "variable"]).reset_index(drop=True)
        print_results_table(df)
        return df

    print("\n수집된 결과가 없습니다.")
    return pd.DataFrame()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="온실 시계열 보간 실험 러너 (block masking)"
    )
    parser.add_argument(
        "--models", type=str, default="autogluon,cafi,linear",
        help="쉼표 구분 모델 키 (autogluon,cafi,linear)",
    )
    parser.add_argument(
        "--loss-rates", type=str, default="0.1,0.3,0.5,0.7,0.9",
        help="쉼표 구분 loss_rate 목록",
    )
    parser.add_argument(
        "--context-len", type=int, default=336,
        help="컨텍스트 길이 (포인트 수, 기본 336=14일)",
    )
    parser.add_argument(
        "--data-dir", type=str, default="../01_data",
        help="PF_*.xlsx 데이터 디렉터리",
    )
    parser.add_argument(
        "--result-dir", type=str, default="../03_result/autogluon",
        help="결과 CSV 저장 디렉터리",
    )
    parser.add_argument(
        "--time-limit", type=int, default=600,
        help="AutoGluon 학습 시간 제한(초, 기본 600=best_quality)",
    )
    parser.add_argument(
        "--save-models", action="store_true", default=True,
        help="AutoGluon predictor와 앙상블 가중치를 result-dir/models/ 에 저장 (기본: 저장)",
    )
    parser.add_argument(
        "--no-save-models", dest="save_models", action="store_false",
        help="모델 저장 비활성화",
    )
    parser.add_argument(
        "--expand-submodels", action="store_true", default=False,
        help="AutoGluon 앙상블 외에 각 sub-model 개별 결과도 results_all.csv에 추가",
    )
    parser.add_argument(
        "--reset", action="store_true", default=False,
        help="기존 results_all.csv를 삭제하고 처음부터 재실행",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    loss_rates = [float(x) for x in args.loss_rates.split(",") if x.strip()]

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (_THIS_DIR / data_dir).resolve()
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = (_THIS_DIR / result_dir).resolve()

    if args.reset:
        csv_path = result_dir / "results_all.csv"
        if csv_path.exists():
            csv_path.unlink()
            print(f"기존 결과 초기화: {csv_path}")

    run_experiment(
        models=models,
        loss_rates=loss_rates,
        context_len=args.context_len,
        data_dir=data_dir,
        result_dir=result_dir,
        time_limit=args.time_limit,
        save_models=args.save_models,
        expand_submodels=args.expand_submodels,
    )


if __name__ == "__main__":
    main()

"""
run_context_ablation.py
Context length 어블레이션 실험

연구 목적:
  Context length가 파운데이션 모델 성능에 미치는 영향을 분석한다.
  원예학적 해석:
    24h  → 일일 관리 패턴 (관수, 환기, 차광)
    48h  → 2일 주기 관리
    96h  → 주간 관리 패턴
    168h → 1주일 (작물 생육 주기)
    336h → 2주 (기존 기본값)
    672h → 4주 (계절 전환 감지)

  각 context length에서: MSE, MAE, NMAE + 추론 시간 → Pareto front 분석용

실행:
  python run_context_ablation.py [--models cafi,autogluon]
                                  [--context-lens 24,48,96,168,336,672]
                                  [--loss-rate 0.3]
                                  [--data-dir ../01_data]
                                  [--result-dir ../03_result/context_ablation]
                                  [--time-limit 60]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

# GPU 호환성 자동 선택 — 반드시 torch import 이전에 위치
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

from preprocessing import preprocess_file, train_test_split, fit_scalers_from_split
from masking import apply_masking
from evaluate import compute_metrics


# ──────────────────────────────────────────────
# 원예학적 해석 메모 (context_len → 의미)
# ──────────────────────────────────────────────
CONTEXT_INTERPRETATION = {
    24: "일일 관리 패턴 (관수/환기/차광)",
    48: "2일 주기 관리",
    96: "주간 관리 패턴",
    168: "1주일 (작물 생육 주기)",
    336: "2주 (기존 기본값)",
    672: "4주 (계절 전환 감지)",
}


# ──────────────────────────────────────────────
# 모델 팩토리
# ──────────────────────────────────────────────
def build_model(model_key: str, context_len: int, time_limit: int = 60):
    """
    model_key와 context_len으로 보간 모델 인스턴스를 생성한다.

    지원 키: cafi | autogluon | timesfm | linear
    """
    key = model_key.strip().lower()

    if key == "cafi":
        from models.foundation_model import CAFIv2Imputation
        return CAFIv2Imputation(
            context_len=context_len,
            max_rounds=5,
            name=f"CAFI-{context_len}h",
        )

    if key == "autogluon":
        # autogluon_model.py는 별도 에이전트가 생성 중이므로 lazy import.
        from models.autogluon_model import AutoGluonImputation
        return AutoGluonImputation(
            context_len=context_len,
            time_limit=time_limit,
        )

    if key == "timesfm":
        from models.foundation_model import TimesFMImputation
        return TimesFMImputation(context_len=context_len)

    if key == "linear":
        from models.linear_interpolation import LinearInterpolation
        # context_len은 적용되지 않지만 baseline으로 포함한다.
        return LinearInterpolation()

    raise ValueError(f"알 수 없는 모델 키: {model_key}")


def _display_name(model_key: str) -> str:
    """결과 표/CSV에 쓸 모델 표시 이름."""
    return {
        "cafi": "CAFI",
        "autogluon": "AutoGluon",
        "timesfm": "TimesFM",
        "linear": "LI",
    }.get(model_key.strip().lower(), model_key)


# ──────────────────────────────────────────────
# 단일 (greenhouse) 데이터 준비
# ──────────────────────────────────────────────
def prepare_greenhouse(filepath: Path, loss_rate: float):
    """
    한 온실 파일을 train/test 분리하고 scaler를 fit한 뒤
    test set에 block 마스킹을 적용한다.

    Returns dict 또는 None(전처리 실패 시).
      name, freq, scalers, ground_truth, masked, artificial_mask, original_mask,
      train_norm
    """
    result = preprocess_file(filepath)
    if result is None:
        return None

    name = result["name"]
    freq = result.get("freq", "1H")

    train, test = train_test_split(result, test_ratio=0.2)
    df_train_raw = train["data_raw"]
    df_test_raw = test["data_raw"]

    # train 통계만으로 scaler fit → 데이터 누수 방지
    scalers, df_train_norm, df_test_norm = fit_scalers_from_split(df_train_raw, df_test_raw)

    if len(df_test_norm) < 10:
        print(f"  [{name}] test set 너무 짧음({len(df_test_norm)}) → skip")
        return None

    # block 마스킹: 어블레이션 동안 loss_rate 고정
    masked, artificial_mask, ground_truth = apply_masking(
        df_test_norm,
        mask_type="block",
        loss_rate=loss_rate,
        freq=freq,
        block_hours=48,
        random_seed=42,
    )

    # 원본 유효 위치(원본 결측은 평가에서 제외)
    original_mask = df_test_norm.notna().astype(float)

    return {
        "name": name,
        "freq": freq,
        "scalers": scalers,
        "ground_truth": ground_truth,
        "masked": masked,
        "artificial_mask": artificial_mask,
        "original_mask": original_mask,
        "train_norm": df_train_norm,
    }


# ──────────────────────────────────────────────
# 메인 실험 루프
# ──────────────────────────────────────────────
def run_ablation(
    models: list[str],
    context_lens: list[int],
    loss_rate: float,
    data_dir: Path,
    result_dir: Path,
    time_limit: int,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("PF_*.xlsx"))
    if not files:
        print(f"데이터 파일 없음: {data_dir}/PF_*.xlsx")
        return pd.DataFrame()

    print("=" * 70)
    print("Context Length Ablation 실험")
    print(f"  models       = {models}")
    print(f"  context_lens = {context_lens}")
    print(f"  loss_rate    = {loss_rate} (고정)")
    print(f"  data_dir     = {data_dir}")
    print(f"  result_dir   = {result_dir}")
    print(f"  time_limit   = {time_limit}s (autogluon)")
    print(f"  파일 수      = {len(files)}")
    print("=" * 70)

    rows: list[dict] = []

    for filepath in files:
        gh = prepare_greenhouse(filepath, loss_rate)
        if gh is None:
            continue

        name = gh["name"]
        scalers = gh["scalers"]
        ground_truth = gh["ground_truth"]
        masked = gh["masked"]
        artificial_mask = gh["artificial_mask"]
        original_mask = gh["original_mask"]
        train_norm = gh["train_norm"]

        for context_len in context_lens:
            interp = CONTEXT_INTERPRETATION.get(context_len, "")
            for model_key in models:
                disp = _display_name(model_key)
                try:
                    model = build_model(model_key, context_len, time_limit=time_limit)

                    # fit
                    t0 = time.perf_counter()
                    model.fit(train_norm)
                    fit_seconds = time.perf_counter() - t0

                    # impute
                    t1 = time.perf_counter()
                    imputed = model.impute(masked, artificial_mask)
                    impute_seconds = time.perf_counter() - t1

                    total_seconds = fit_seconds + impute_seconds

                    metrics = compute_metrics(
                        ground_truth, imputed, original_mask, artificial_mask, scalers
                    )

                    # 변수별 행 저장
                    maes = []
                    for variable, m in metrics.items():
                        rows.append({
                            "model": disp,
                            "model_key": model_key.strip().lower(),
                            "greenhouse": name,
                            "context_len": context_len,
                            "context_meaning": interp,
                            "loss_rate": loss_rate,
                            "variable": variable,
                            "MSE": m["MSE"],
                            "MAE": m["MAE"],
                            "NMAE": m["NMAE"],
                            "n_eval": m["n_eval"],
                            "fit_seconds": fit_seconds,
                            "impute_seconds": impute_seconds,
                            "total_seconds": total_seconds,
                        })
                        if not np.isnan(m["MAE"]):
                            maes.append(m["MAE"])

                    mean_mae = float(np.mean(maes)) if maes else float("nan")
                    print(
                        f"context_len={context_len}, model={disp}, greenhouse={name}: "
                        f"MAE={mean_mae:.5f} "
                        f"(fit={fit_seconds:.1f}s, impute={impute_seconds:.1f}s)"
                    )

                except Exception as e:
                    print(
                        f"[ERROR] context_len={context_len}, model={disp}, "
                        f"greenhouse={name}: {e}"
                    )
                    traceback.print_exc()
                    continue

    if not rows:
        print("\n수집된 결과가 없습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── 변수별 결과 저장 ──
    results_path = result_dir / "ablation_results.csv"
    df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"\n변수별 결과 저장: {results_path}  ({len(df)} rows)")

    # ── context_len별 집계 (온실·변수 평균) ──
    summary = (
        df.groupby(["model", "model_key", "context_len", "loss_rate"], as_index=False)
        .agg(
            MSE=("MSE", "mean"),
            MAE=("MAE", "mean"),
            NMAE=("NMAE", "mean"),
            fit_seconds=("fit_seconds", "mean"),
            impute_seconds=("impute_seconds", "mean"),
            total_seconds=("total_seconds", "mean"),
            n_eval=("n_eval", "sum"),
        )
        .sort_values(["model", "context_len"])
    )
    summary_path = result_dir / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"집계 결과 저장: {summary_path}  ({len(summary)} rows)")

    _print_table(summary)

    return df


def _print_table(summary: pd.DataFrame) -> None:
    """context_len vs model 의 mean_MAE 테이블 출력."""
    print("\n" + "=" * 70)
    print("[ context_len x model : mean MAE (온실·변수 평균) ]")
    print("=" * 70)

    try:
        pivot = summary.pivot_table(
            index="context_len", columns="model", values="MAE", aggfunc="mean"
        )
        pivot = pivot.sort_index()
        print(pivot.round(5).to_string())
    except Exception:
        # pivot 실패 시 long 형태로 출력
        for _, r in summary.iterrows():
            print(f"  context_len={r['context_len']:>4} | {r['model']:<10} | "
                  f"MAE={r['MAE']:.5f} | impute={r['impute_seconds']:.1f}s")
    print("=" * 70)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Context length 어블레이션 실험")
    p.add_argument("--models", default="cafi,autogluon",
                   help="콤마 구분 모델 키 (cafi,autogluon,timesfm,linear)")
    p.add_argument("--context-lens", default="24,48,96,168,336,672",
                   help="콤마 구분 context length(시간) 정수 리스트")
    p.add_argument("--loss-rate", type=float, default=0.3,
                   help="블록 마스킹 결측률 (어블레이션 동안 고정)")
    p.add_argument("--data-dir", default="../01_data",
                   help="PF_*.xlsx 가 있는 디렉터리")
    p.add_argument("--result-dir", default="../03_result/context_ablation",
                   help="결과 CSV 저장 디렉터리")
    p.add_argument("--time-limit", type=int, default=60,
                   help="AutoGluon time_limit (초)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    context_lens = [int(c.strip()) for c in args.context_lens.split(",") if c.strip()]

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (_THIS_DIR / data_dir).resolve()
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = (_THIS_DIR / result_dir).resolve()

    run_ablation(
        models=models,
        context_lens=context_lens,
        loss_rate=args.loss_rate,
        data_dir=data_dir,
        result_dir=result_dir,
        time_limit=args.time_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
run_anomaly.py
모델 잔차 기반 스파이크 탐지 + 합성 주입 검증/복원 실행 (이상치탐지_계획.md 구현)

범위(사용자 확정): 이 온실 데이터에는 드리프트/고착이 없다고 보고 **스파이크만**
탐지한다. Test 온실에 합성 스파이크를 주입해 "탐지 → 결측 표시 → 같은 모델로
복원"까지 한 번에 검증한다(원래 값을 알고 있으므로 복원 정확도까지 측정 가능).

run_comparison.py가 만든 체크포인트(03_result/comparison/models/)를 재사용해
Test 온실에서:
  1. 여러 모델로 잔차 스캔 → 변수별 MAD z-score(|z|>tau)로 스파이크 flag
     → 다중모델 합의(consensus) 점수.
  2. 합성 스파이크 주입 → 탐지(precision/recall/F1, tau 스윕) → 복원(MAE) 검증.
  3. (실데이터) 합의 임계 이상 위치를 결측으로 표시 후 같은 모델로 복원(정제본 저장).

사전 조건: run_comparison.py를 먼저 실행해 모델 체크포인트가 있어야 한다
(zero-shot 모델은 체크포인트 없이도 즉시 사용 가능).

실행:
  python run_anomaly.py                  # Test 전체, 기본 모델셋
  python run_anomaly.py --quick          # 스모크: 1온실 · CAFI만 · 큰 block
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import gpu_utils  # noqa: F401, E402

import numpy as np
import pandas as pd

from preprocessing import preprocess_file, TARGET_COLS
from data_split import load_split
from checkpoint import CheckpointManager
from evaluate import _inverse_transform
from anomaly_detection import (
    scan_residuals,
    mad_zscore,
    spike_flag,
    consensus_score,
    inject_synthetic_spikes,
    evaluate_detection,
    restoration_error,
    restore_flagged,
)

TARGET_VARS = list(TARGET_COLS.values())
DEFAULT_MODELS = ["LI", "SeasonalNaive", "AG-LightGBM", "AG-RandomForest",
                  "AG-DeepAR", "AG-PatchTST", "Chronos2", "TimesFM2.5", "CAFI"]
# F7b τ-sweep 결과 적정 τ가 ~18~21로 확인되어 범위를 상향(기존 2~5는 전부 과탐 구간).
TAU_SWEEP = [3, 6, 9, 12, 15, 18, 21, 25]


# ──────────────────────────────────────────────────────────────────────────────
# 모델 로드 (run_comparison.py 체크포인트 재사용, 재학습 없음)
# ──────────────────────────────────────────────────────────────────────────────

def _load_models(model_keys: list[str], ckpt: CheckpointManager, context_len: int) -> dict[str, object]:
    from models.foundation_model import ChronosImputation, TimesFMImputation, CAFIImputation
    from models.linear_interpolation import LinearInterpolation
    from models.seasonal_naive import SeasonalNaiveImputation

    models: dict[str, object] = {}
    for k in model_keys:
        if k == "LI":
            models[k] = LinearInterpolation()
        elif k == "SeasonalNaive":
            models[k] = SeasonalNaiveImputation(season_length=24, context_len=context_len)
        elif k == "CAFI":
            # CAFI v1 (Chronos covariate, zero-shot) — 보간 파이프라인과 동일 버전으로 통일.
            # zero-shot이라 체크포인트가 불필요하다.
            models[k] = CAFIImputation(fine_tune=False, context_len=context_len, name="CAFI")
        elif k.startswith("AG-"):
            try:
                from models.autogluon_model import AutoGluonImputation
            except Exception as e:
                warnings.warn(f"[{k}] autogluon 로드 실패 → skip: {e}")
                continue
            mdir = ckpt.model_dir(k)
            if not ckpt.autogluon_exists(k):
                warnings.warn(f"[{k}] 체크포인트 없음 → skip (run_comparison.py 먼저 실행)")
                continue
            models[k] = AutoGluonImputation.load(mdir, context_len=context_len, time_limit=600)
        elif k == "Chronos2":
            models[k] = ChronosImputation(context_len=context_len, fine_tune=False)
        elif k == "TimesFM2.5":
            models[k] = TimesFMImputation(context_len=context_len)
        else:
            warnings.warn(f"알 수 없는 모델 키 → skip: {k}")
    return models


def _model_input_frame(
    model_name: str, full_with_cov: pd.DataFrame, target_cols: list[str], covariate_cols: list[str]
) -> pd.DataFrame:
    """CAFI만 covariate를 함께 전달(슈퍼셋이면 impute() 내부 가드가 안전하게 재정렬)."""
    if model_name == "CAFI" and covariate_cols:
        return full_with_cov[target_cols + covariate_cols]
    return full_with_cov[target_cols]


# ──────────────────────────────────────────────────────────────────────────────
# 단일 온실: 실데이터 스파이크 탐지 + 합의
# ──────────────────────────────────────────────────────────────────────────────

def _detect_one_greenhouse(
    greenhouse: str,
    data: pd.DataFrame,
    target_cols: list[str],
    covariate_cols: list[str],
    models: dict[str, object],
    block_h: int,
    tau: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """모델별 스파이크 flag를 계산해 timeline과 다중모델 합의(consensus) DataFrame을 반환한다."""
    flag_by_model: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []

    for mname, model in models.items():
        frame = _model_input_frame(mname, data, target_cols, covariate_cols)
        residual = scan_residuals(model, frame, target_cols, block_h=block_h)

        flags = pd.DataFrame(False, index=data.index, columns=target_cols)
        for col in target_cols:
            z = mad_zscore(residual[col])
            flag = spike_flag(z, tau=tau)
            flags[col] = flag
            for t in data.index[flag]:
                rows.append({
                    "greenhouse": greenhouse, "model": mname, "variable": col,
                    "timestamp": t, "z": float(z.loc[t]) if pd.notna(z.loc[t]) else np.nan,
                })
        flag_by_model[mname] = flags

    timeline = pd.DataFrame(rows, columns=["greenhouse", "model", "variable", "timestamp", "z"])

    # ── 합의 점수: 모델별 flag(bool)를 변수마다 평균 ──
    consensus_rows: list[dict] = []
    for col in target_cols:
        per_model_flags = {m: flag_by_model[m][col] for m in models}
        cons = consensus_score(per_model_flags)
        flagged = cons >= 0.5
        for t in data.index[flagged]:
            consensus_rows.append({
                "greenhouse": greenhouse, "variable": col, "timestamp": t,
                "consensus_score": float(cons.loc[t]),
            })
    consensus_df = pd.DataFrame(consensus_rows, columns=["greenhouse", "variable", "timestamp", "consensus_score"])

    return timeline, consensus_df


# ──────────────────────────────────────────────────────────────────────────────
# 단일 온실: 합성 스파이크 주입 → 탐지 → 복원 (before/after 데모)
# ──────────────────────────────────────────────────────────────────────────────

def _inject_detect_restore(
    greenhouse: str,
    data: pd.DataFrame,
    target_cols: list[str],
    covariate_cols: list[str],
    models: dict[str, object],
    block_h: int,
    tau: float,
    n_inject: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    클린 데이터에 합성 스파이크를 주입 → 모델로 탐지(tau 스윕 P/R/F1) →
    --tau 기준으로 복원 → 복원값을 원래 진짜 값과 비교(before/after).

    Returns
    -------
    (synth_metrics: greenhouse/variable/model/tau별 precision·recall·f1,
     demo: greenhouse/variable/model/timestamp별 true/corrupted/restored 값 + 오차)
    """
    metric_rows: list[dict] = []
    demo_rows: list[dict] = []
    valid_mask = data.notna()

    for col in target_cols:
        injected_series, injected_pos = inject_synthetic_spikes(
            data[col], valid_mask[col], n_inject=n_inject, seed=seed,
        )
        if injected_pos.empty:
            continue
        corrupted = data.copy()
        corrupted[col] = injected_series

        for mname, model in models.items():
            frame = _model_input_frame(mname, corrupted, target_cols, covariate_cols)
            residual = scan_residuals(model, frame, target_cols, block_h=block_h)
            z = mad_zscore(residual[col])

            for t in TAU_SWEEP:
                flag = spike_flag(z, tau=t)
                metrics = evaluate_detection(flag, injected_pos, n=len(data))
                metric_rows.append({
                    "greenhouse": greenhouse, "variable": col, "model": mname, "tau": t,
                    **metrics,
                })

            # ── 운영 tau로 실제 복원(before/after) 수행 ──
            flag = spike_flag(z, tau=tau)
            flagged_mask = pd.DataFrame(False, index=data.index, columns=target_cols)
            flagged_mask[col] = flag
            restored_full = restore_flagged(model, frame, target_cols, flagged_mask)
            restored_col = restored_full[col]

            r_err = restoration_error(restored_col, injected_pos)
            for _, row in injected_pos.iterrows():
                t_idx = int(row["start"])
                ts = data.index[t_idx]
                demo_rows.append({
                    "greenhouse": greenhouse, "variable": col, "model": mname, "timestamp": ts,
                    "true_value": row["true_value"],
                    "corrupted_value": float(injected_series.iloc[t_idx]),
                    "detected": bool(flag.iloc[t_idx]),
                    "restored_value": float(restored_col.iloc[t_idx]),
                    "abs_error_after_restore": abs(float(restored_col.iloc[t_idx]) - row["true_value"]),
                })
            print(f"    [{mname}] {col}: 주입 {r_err['n']}개 / 복원 MAE={r_err['mae']:.4f} (tau={tau})")

    return pd.DataFrame(metric_rows), pd.DataFrame(demo_rows)


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def run_anomaly(
    comparison_dir: Path,
    result_dir: Path,
    model_keys: list[str],
    context_len: int,
    block_h: int,
    tau: float,
    n_inject: int,
    seed: int,
    restore_model: str,
    n_greenhouses: int | None,
) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "cleaned").mkdir(parents=True, exist_ok=True)

    split = load_split(comparison_dir)
    test_paths = [Path(p) for p in split["test"]]
    if n_greenhouses:
        test_paths = test_paths[:n_greenhouses]

    ckpt = CheckpointManager(comparison_dir / "models")
    models = _load_models(model_keys, ckpt, context_len)
    if not models:
        raise RuntimeError(
            "로드된 모델이 없습니다. run_comparison.py를 먼저 실행해 체크포인트를 만드세요."
        )
    if restore_model not in models:
        warnings.warn(f"--restore-model {restore_model}이 로드된 모델에 없음 → 복원 단계 건너뜀")
        restore_model = ""

    print("=" * 72)
    print("모델 잔차 기반 스파이크 탐지 (드리프트/고착 제외)")
    print(f"  Test 온실: {len(test_paths)} | 모델: {list(models.keys())}")
    print(f"  block_h={block_h}h | tau={tau} | n_inject={n_inject} | restore={restore_model or '(없음)'}")
    print("=" * 72)

    all_timeline, all_consensus, all_synth, all_demo = [], [], [], []

    for fp in test_paths:
        res = preprocess_file(fp, include_covariates=True)
        if res is None:
            continue
        gh = res["name"]
        data = res["data"]
        scalers = res["scaler"]
        target_cols = [c for c in TARGET_VARS if c in data.columns]
        if len(target_cols) < len(TARGET_VARS):
            print(f"  {gh}: 타깃 변수 부족 → skip")
            continue
        covariate_cols = [c for c in data.columns if c not in TARGET_VARS]

        print(f"\n[{gh}] 실데이터 스파이크 스캔 중...")
        timeline, consensus_df = _detect_one_greenhouse(
            gh, data, target_cols, covariate_cols, models, block_h, tau,
        )
        print(f"  [{gh}] flag 발생: {len(timeline)}행 | 합의(>=0.5) {len(consensus_df)}행")
        all_timeline.append(timeline)
        all_consensus.append(consensus_df)

        print(f"  [{gh}] 합성 스파이크 주입 → 탐지 → 복원 검증 중...")
        synth, demo = _inject_detect_restore(
            gh, data, target_cols, covariate_cols, models, block_h, tau, n_inject, seed,
        )
        all_synth.append(synth)
        all_demo.append(demo)

        # ── 실데이터 복원: 합의(>=0.5) 위치를 결측 표시 후 같은 모델로 보간 ──
        if restore_model:
            flagged_mask = pd.DataFrame(False, index=data.index, columns=target_cols)
            for _, row in consensus_df.iterrows():
                flagged_mask.loc[row["timestamp"], row["variable"]] = True
            frame = _model_input_frame(restore_model, data, target_cols, covariate_cols)
            restored = restore_flagged(models[restore_model], frame, target_cols, flagged_mask)
            restored_phys = _inverse_transform(restored[target_cols], scalers)
            restored_phys.to_csv(result_dir / "cleaned" / f"{gh}.csv", encoding="utf-8-sig")
            print(f"  [{gh}] 정제본 저장: {result_dir / 'cleaned' / (gh + '.csv')}")

    timeline_df = pd.concat(all_timeline, ignore_index=True) if all_timeline else pd.DataFrame()
    consensus_df_all = pd.concat(all_consensus, ignore_index=True) if all_consensus else pd.DataFrame()
    synth_df = pd.concat(all_synth, ignore_index=True) if all_synth else pd.DataFrame()
    demo_df = pd.concat(all_demo, ignore_index=True) if all_demo else pd.DataFrame()

    timeline_df.to_csv(result_dir / "timeline.csv", index=False, encoding="utf-8-sig")
    consensus_df_all.to_csv(result_dir / "consensus.csv", index=False, encoding="utf-8-sig")
    synth_df.to_csv(result_dir / "synthetic_validation.csv", index=False, encoding="utf-8-sig")
    demo_df.to_csv(result_dir / "injection_demo.csv", index=False, encoding="utf-8-sig")

    print(f"\n{'='*72}")
    print(f"완료. timeline={len(timeline_df)}행, consensus={len(consensus_df_all)}행, "
          f"synthetic_validation={len(synth_df)}행, injection_demo={len(demo_df)}행 → {result_dir}")
    if not synth_df.empty:
        print("\n[합성 스파이크 검증 — 모델×tau 평균 F1]")
        try:
            tbl = synth_df.groupby(["model", "tau"])["f1"].mean().round(3).unstack("tau")
            print(tbl.to_string())
        except Exception as e:
            print(f"  요약 실패: {e}")
    if not demo_df.empty:
        print(f"\n[복원 정확도 — tau={tau} 기준, 모델별 평균 abs_error]")
        try:
            print(demo_df.groupby("model")["abs_error_after_restore"].mean().round(4).to_string())
        except Exception as e:
            print(f"  요약 실패: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="모델 잔차 기반 스파이크 탐지 + 합성 주입 검증/복원")
    p.add_argument("--comparison-dir", type=str, default="../03_result/comparison",
                   help="run_comparison.py의 결과 디렉터리(split.json·체크포인트 위치)")
    p.add_argument("--result-dir", type=str, default="../03_result/anomaly")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument("--context-len", type=int, default=720)
    p.add_argument("--block-h", type=int, default=6, help="잔차 스캔 블록 크기(시간)")
    p.add_argument("--tau", type=float, default=3.5, help="스파이크 판정 |z| 임계")
    p.add_argument("--n-inject", type=int, default=20, help="합성 스파이크 주입 개수(변수당)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restore-model", type=str, default="CAFI",
                   help="합의 이상치를 복원할 모델(빈 문자열이면 복원 생략)")
    p.add_argument("--n-greenhouses", type=int, default=None, help="처리할 Test 온실 수 제한")
    p.add_argument("--quick", action="store_true",
                   help="스모크: 1온실 · CAFI만 · block 24h · n_inject 5")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    block_h, n_inject, n_gh = args.block_h, args.n_inject, args.n_greenhouses

    if args.quick:
        model_keys = ["CAFI"]
        block_h = 24
        n_inject = 5
        n_gh = 1
        print("[--quick] 스모크 모드: Test 1온실 · CAFI만 · block 24h · n_inject 5")

    comparison_dir = Path(args.comparison_dir)
    result_dir = Path(args.result_dir)
    if not comparison_dir.is_absolute():
        comparison_dir = (_THIS_DIR / comparison_dir).resolve()
    if not result_dir.is_absolute():
        result_dir = (_THIS_DIR / result_dir).resolve()

    run_anomaly(
        comparison_dir=comparison_dir, result_dir=result_dir,
        model_keys=model_keys, context_len=args.context_len, block_h=block_h,
        tau=args.tau, n_inject=n_inject, seed=args.seed,
        restore_model=args.restore_model, n_greenhouses=n_gh,
    )


if __name__ == "__main__":
    main()

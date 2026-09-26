"""
run_comparison.py
온실 결측복원 10개 방법론 통합 비교 (개선사항.md #4, #5)

교차-온실 분할(data_split) 위에서 9개 모델을 1회 학습 후 재사용(checkpoint)하여
Test 온실에 인위 갭을 생성하고 보간 성능을 계절×시간대로 층화 평가한다.

비교 라인업:
  LI, SeasonalNaive, AG-LightGBM, AG-RandomForest, AG-DeepAR(LSTM), AG-PatchTST,
  Chronos2, TimesFM2.5, TimesFM3.0, CAFI(covariate 확장)

결과: 03_result/comparison/results.csv (계절×시간대 층화 지표, append/resume)

실행:
  python run_comparison.py                 # 전체
  python run_comparison.py --quick         # 스모크 (Test 2온실·gap 24h·1 repeat)
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import warnings
import zlib
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# GPU 호환성: torch import 이전에 실행
import gpu_utils  # noqa: F401

import numpy as np
import pandas as pd

from preprocessing import preprocess_file, TARGET_COLS
from data_split import (
    build_cross_greenhouse_split,
    load_split,
    SUMMER_MONTHS,
    WINTER_MONTHS,
)
from masking_v2 import create_gap_masks, SCENARIO_CONFIGS, RANDOM_SEED
from evaluate import compute_all_metrics
from checkpoint import CheckpointManager

from models.linear_interpolation import LinearInterpolation
from models.seasonal_naive import SeasonalNaiveImputation
from models.foundation_model import (
    ChronosImputation,
    TimesFMImputation,
    TimesFM3Imputation,
    TimesFM3MVImputation,
    TimesFM3CovImputation,
    CAFIImputation,
    CAFITimesFM3Imputation,
)

TARGET_VARS = list(TARGET_COLS.values())  # ['Tin','Tout','RH','CO2','Rad']
_NEIGHBOR_BANK = None


def _stable_seed(text: str) -> int:
    """프로세스 간 재현 가능한 시드 오프셋. 내장 hash()는 PYTHONHASHSEED salt로
    매 프로세스 달라져 마스크 위치가 재현되지 않으므로 zlib.crc32로 고정한다."""
    return zlib.crc32(text.encode("utf-8")) % 9999

# 표시명 → (구현 종류, 빌더 설정). AutoGluon 매핑은 사용자 확정값.
#   LightGBM=RecursiveTabular(GBM), RF=RecursiveTabular(RF), LSTM=DeepAR, PatchTST=PatchTST
AG_SPECS = {
    "AG-LightGBM":     {"RecursiveTabular": {"model_name": "GBM"}},
    "AG-RandomForest": {"RecursiveTabular": {"model_name": "RF"}},
    "AG-DeepAR":       {"DeepAR": {}},
    "AG-PatchTST":     {"PatchTST": {}},
}

_CSV_COLS = [
    "greenhouse", "model", "scenario", "masked_vars", "gap_length_h", "repeat",
    "group_type", "group_value", "variable",
    "MSE", "MAE", "NMAE", "n_eval", "context_len",
]


# ──────────────────────────────────────────────────────────────────────────────
# 그룹 라벨 (계절 × 시간대)
# ──────────────────────────────────────────────────────────────────────────────

def make_group_labels(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """계절·시간대·계절×시간대 그룹 라벨을 생성한다(개선사항.md #4 층화 분석)."""
    months = pd.Index(index).month
    hours = pd.Index(index).hour
    season = np.where(np.isin(months, list(SUMMER_MONTHS)), "summer",
              np.where(np.isin(months, list(WINTER_MONTHS)), "winter",
              np.where(np.isin(months, [3, 4, 5]), "spring", "fall")))
    tod = np.where(hours < 12, "AM", "PM")
    season_s = pd.Series(season, index=index)
    tod_s = pd.Series(tod, index=index)
    season_tod = pd.Series([f"{s}_{t}" for s, t in zip(season, tod)], index=index)
    return {"season": season_s, "time_of_day": tod_s, "season_tod": season_tod}


# ──────────────────────────────────────────────────────────────────────────────
# 모델 빌드 + 1회 학습/재사용
# ──────────────────────────────────────────────────────────────────────────────

def build_and_train_models(
    model_keys: list[str],
    context_len: int,
    ckpt: CheckpointManager,
    train_frames_target: list[pd.DataFrame],
    train_frames_cov: list[pd.DataFrame],
    train_names: list[str],
    ag_time_limit: int,
    train_paths: list[Path] | None = None,
    validation_frames: list[pd.DataFrame] | None = None,
) -> dict[str, object]:
    """
    모델 인스턴스를 생성하고, 학습형 모델은 체크포인트가 있으면 로드, 없으면 학습/저장.

    train_frames_target : 학습 온실 5변수 프레임(AutoGluon 글로벌 학습용)
    train_frames_cov    : 학습 온실 5변수+covariate 프레임(CAFI 글로벌 학습용)
    """
    global _NEIGHBOR_BANK
    models: dict[str, object] = {}

    for key in model_keys:
        k = key.strip()

        if k == "LI":
            models[k] = LinearInterpolation()

        elif k == "SeasonalNaive":
            models[k] = SeasonalNaiveImputation(season_length=24, context_len=context_len)

        elif k in AG_SPECS:
            try:
                from models.autogluon_model import AutoGluonImputation
            except Exception as e:
                warnings.warn(f"[{k}] autogluon 로드 실패 → skip: {e}")
                continue
            mdir = ckpt.model_dir(k)
            init = dict(context_len=context_len, time_limit=ag_time_limit,
                        hyperparameters=AG_SPECS[k], name=k, save_path=mdir)
            if ckpt.autogluon_exists(k):
                try:
                    models[k] = AutoGluonImputation.load(mdir, **init)
                    print(f"  [{k}] predictor 로드 (학습 생략)")
                    continue
                except Exception as e:
                    warnings.warn(f"[{k}] 로드 실패 → 재학습: {e}")
            m = AutoGluonImputation(**init)
            t0 = time.perf_counter()
            m.fit_global(train_frames_target, names=train_names, validation_frames=validation_frames)
            print(f"  [{k}] 글로벌 학습 완료 ({time.perf_counter()-t0:.1f}s)")
            ckpt.record(k, "autogluon", {"n_train_gh": len(train_frames_target)})
            models[k] = m

        elif k == "Chronos2":
            models[k] = ChronosImputation(context_len=context_len, fine_tune=False)

        elif k == "TimesFM2.5":
            models[k] = TimesFMImputation(context_len=context_len)

        elif k == "TimesFM3.0":
            # TimesFM 3.0 zero-shot. 2.5와 동일한 context/horizon으로 head-to-head 비교.
            models[k] = TimesFM3Imputation(context_len=context_len)

        elif k == "TimesFM3.0-MV":
            # 다변량(variate attention): 변수 간 정보가 과거 구간으로만 흐른다.
            models[k] = TimesFM3MVImputation(context_len=context_len)

        elif k == "TimesFM3.0-COV":
            # past_future_covariates: gap 구간 타 변수값을 조건으로 사용(= CAFI 1-pass 등가).
            models[k] = TimesFM3CovImputation(context_len=context_len)

        elif k in ("Spatial-Ridge", "TimesFM3.0-COV-SPA"):
            # 공간축(cross-site) arm — 이웃은 train 온실로만 제한한다.
            if not train_paths:
                warnings.warn(f"[{k}] train_paths 없음 → skip")
                continue
            import spatial as _sp
            if _NEIGHBOR_BANK is None:
                _NEIGHBOR_BANK = _sp.NeighborBank(train_paths, TARGET_VARS)
            if k == "Spatial-Ridge":
                models[k] = _sp.SpatialRidgeImputation(_NEIGHBOR_BANK)
            else:
                models[k] = _sp.make_tfm3_cov_spatial(_NEIGHBOR_BANK, context_len)
            print(f"  [{k}] 공간축 준비 (이웃 후보 = train 온실 {len(_NEIGHBOR_BANK.sites)}개)")

        elif k in ("BiTFI-TimesFM3", "BiTFI-Chronos2", "BiTFI-TimesFM3-fwd"):
            # BiTFI — Bidirectional Time-series Foundation-model Imputation (양방향 융합 및 공변량 보정).
            if not train_paths:
                warnings.warn(f"[{k}] train_paths 없음 → skip"); continue
            import spatial as _sp
            import bitfi as _bitfi
            if _NEIGHBOR_BANK is None:
                _NEIGHBOR_BANK = _sp.NeighborBank(train_paths, TARGET_VARS)
            if k == "BiTFI-Chronos2":
                models[k] = _bitfi.BiTFIChronos2(bank=_NEIGHBOR_BANK, context_len=context_len, name=k)
            else:
                models[k] = _bitfi.BiTFITimesFM3(bank=_NEIGHBOR_BANK, context_len=context_len, name=k,
                                               bidirectional=(k != "BiTFI-TimesFM3-fwd"))
            print(f"  [{k}] 준비 (bidirectional={getattr(models[k], 'bidirectional', None)})")

        elif k in ("CAFI-R1", "CAFI-TimesFM3-R1"):
            # 라운드별 오차 추적 결과(시나리오 C에서 R1 최적, R2+ 단조 악화)에 따라
            # 반복을 1회로 고정한 CAFI. A·B는 5라운드 버전과 동일, C만 달라진다.
            if k == "CAFI-R1":
                models[k] = CAFIImputation(fine_tune=False, context_len=context_len,
                                           max_rounds=1, name=k)
            else:
                models[k] = CAFITimesFM3Imputation(context_len=context_len,
                                                   max_rounds=1, name=k)
            models[k].uses_ext_covariates = False
            print(f"  [{k}] CAFI max_rounds=1 준비")

        elif k == "CAFI-TimesFM3":
            # CAFI 알고리즘 그대로, 백본만 Chronos-2 → TimesFM 3.0으로 교체.
            models[k] = CAFITimesFM3Imputation(context_len=context_len, name=k)
            models[k].uses_ext_covariates = False
            print(f"  [{k}] CAFI (TimesFM-3.0 백본, zero-shot) 준비")

        elif k == "CAFI":
            # CAFI v1: Chronos-2 네이티브 covariate iterative refinement (zero-shot).
            # v2(TimesFM+글로벌 선형 Ridge)는 Ridge가 온실 간 전이가 안 돼 CO2/RH에서
            # 정확도가 붕괴함을 실측 확인 → covariate-awareness가 실제로 작동하는 v1으로 복원.
            # zero-shot이라 글로벌 학습/체크포인트가 불필요하다.
            models[k] = CAFIImputation(fine_tune=False, context_len=context_len, name="CAFI")
            # 최종 CAFI는 외부 환경 covariate(WindSpeed/RadIn)를 쓰지 않는다:
            # 변수 간 cross-variable refinement + 캘린더 covariate만 사용해 모든
            # 모델과 동일한 5개 타깃만 입력으로 받는다(공정 비교, leakage 방지).
            models[k].uses_ext_covariates = False
            print(f"  [{k}] CAFI v1 (cross-variable refinement, no external covariates, zero-shot) 준비")

        else:
            warnings.warn(f"알 수 없는 모델 키 → skip: {k}")

    return models


# ──────────────────────────────────────────────────────────────────────────────
# 중간 저장 / 재개
# ──────────────────────────────────────────────────────────────────────────────

def _load_completed(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(zip(
            df["greenhouse"], df["model"], df["scenario"],
            df["masked_vars"], df["gap_length_h"], df["repeat"],
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
# 한 모델 × 한 마스크 평가 → 결과 행 생성
# ──────────────────────────────────────────────────────────────────────────────

def _eval_one(
    model_name, model, mask_result, scalers, group_labels,
    greenhouse, covariate_cols, target_frame_cols, context_len,
    base=None, norm_ranges=None,
) -> list[dict]:
    """모델을 실행하고 계절×시간대 층화 + 전체 지표 행을 만든다."""
    # CAFI만 covariate 컬럼을 입력에 포함한다(나머지는 5변수 타깃 프레임).
    # ablation variant는 인스턴스 속성 uses_ext_covariates로 제어(기본 CAFI는 이름으로 fallback).
    uses_ext_cov = getattr(model, "uses_ext_covariates", model_name == "CAFI")
    if uses_ext_cov and covariate_cols:
        cov_obs = mask_result.ground_truth[covariate_cols]
        masked_in = pd.concat([mask_result.masked_data[target_frame_cols], cov_obs], axis=1)
        ones = pd.DataFrame(1.0, index=cov_obs.index, columns=covariate_cols)
        mask_in = pd.concat([mask_result.effective_mask[target_frame_cols], ones], axis=1)
    else:
        masked_in = mask_result.masked_data[target_frame_cols]
        mask_in = mask_result.effective_mask[target_frame_cols]

    # base 캐시가 있으면(원본결측 1회 보간 재사용) 인공 gap만 재예측하는 빠른 경로.
    # 평가 위치 결과는 기존 impute()와 동일하다(컬럼 독립 + 컬럼당 인공 gap 1개).
    if base is not None and hasattr(model, "impute_artificial"):
        artificial_bool = mask_result.artificial_mask[list(masked_in.columns)] == 0
        imputed_full = model.impute_artificial(masked_in, mask_in, artificial_bool, base)
    else:
        imputed_full = model.impute(masked_in, mask_in)
    # 평가는 항상 5개 타깃 컬럼에 대해서만 수행한다.
    imputed = imputed_full[target_frame_cols]
    gt = mask_result.ground_truth[target_frame_cols]
    art = mask_result.artificial_mask[target_frame_cols]
    # original_valid_mask 복원: effective = original * artificial 이므로
    # artificial==1 위치에서는 effective가 곧 original, artificial==0(인위결측) 위치는
    # 평가에서 original=1로 둔다(원본 유효 위치만 eval_mask로 걸러짐).
    orig = pd.DataFrame(
        np.where(art.values == 1, mask_result.effective_mask[target_frame_cols].values, 1.0),
        index=art.index, columns=target_frame_cols,
    )

    target_scalers = {c: scalers.get(c) for c in target_frame_cols}
    overall, subgroup = compute_all_metrics(
        ground_truth=gt, imputed=imputed,
        original_mask=orig, artificial_mask=art,
        group_labels=group_labels, scalers=target_scalers,
        norm_ranges=norm_ranges,
    )

    vars_key = ",".join(mask_result.masked_vars)
    rows: list[dict] = []
    base = dict(greenhouse=greenhouse, model=model_name, scenario=mask_result.scenario,
                masked_vars=vars_key, gap_length_h=mask_result.gap_length_h,
                repeat=mask_result.repeat, context_len=context_len)

    # 전체(group_type='all')
    for var, m in overall.items():
        if m["n_eval"] == 0:
            continue
        rows.append({**base, "group_type": "all", "group_value": "all", "variable": var,
                     "MSE": _r(m["MSE"]), "MAE": _r(m["MAE"]), "NMAE": _r(m.get("NMAE")),
                     "n_eval": m["n_eval"]})
    # 층화
    for gtype, gvals in subgroup.items():
        for gval, varmap in gvals.items():
            for var, m in varmap.items():
                if m["n_eval"] == 0:
                    continue
                rows.append({**base, "group_type": gtype, "group_value": gval, "variable": var,
                             "MSE": _r(m["MSE"]), "MAE": _r(m["MAE"]), "NMAE": _r(m.get("NMAE")),
                             "n_eval": m["n_eval"]})
    return rows


def _r(x):
    return round(float(x), 6) if x is not None and not (isinstance(x, float) and np.isnan(x)) else np.nan


# ──────────────────────────────────────────────────────────────────────────────
# 메인 비교 루프
# ──────────────────────────────────────────────────────────────────────────────

def run_comparison(
    data_dir: Path,
    result_dir: Path,
    model_keys: list[str],
    scenarios: list[str],
    gap_lengths_h: list[int],
    n_repeats: int,
    context_len: int,
    n_test: int,
    seed: int,
    ag_time_limit: int,
    reset: bool = False,
    rebuild_split: bool = False,
    reset_models: list[str] | None = None,
) -> pd.DataFrame:
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "results.csv"
    if reset and csv_path.exists():
        csv_path.unlink()
        print(f"기존 결과 초기화: {csv_path}")

    # 특정 모델만 재계산: 해당 모델 행을 results.csv에서 제거(백업 후) → resume이 다시 계산한다.
    if reset_models and csv_path.exists():
        try:
            _df = pd.read_csv(csv_path)
            backup = csv_path.with_name("results_before_reset.csv")
            _df.to_csv(backup, index=False, encoding="utf-8-sig")
            keep = _df[~_df["model"].isin(reset_models)]
            keep.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"reset-models {reset_models}: {len(_df) - len(keep)}행 제거 "
                  f"(백업 → {backup.name}, 남은 {len(keep)}행)")
        except Exception as e:
            warnings.warn(f"reset-models 처리 실패: {e}")

    ckpt = CheckpointManager(result_dir / "models")

    # ── 1) 분할 ──
    split_path = result_dir / "split.json"
    if rebuild_split or not split_path.exists():
        split = build_cross_greenhouse_split(data_dir, n_test=n_test, seed=seed, result_dir=result_dir)
    else:
        split = load_split(result_dir)
        print(f"기존 split.json 재사용 (train {len(split['train'])} / test {len(split['test'])})")

    train_paths = [Path(p) for p in split["train"]]
    test_paths = [Path(p) for p in split["test"]][:n_test] if n_test else [Path(p) for p in split["test"]]

    print("=" * 72)
    print("온실 결측복원 10개 방법론 통합 비교 (교차-온실 분할)")
    print(f"  모델: {model_keys}")
    print(f"  Train 온실: {len(train_paths)} | Test 온실: {len(test_paths)}")
    print(f"  시나리오: {scenarios} | Gap: {gap_lengths_h}h | repeat≤{n_repeats} | ctx={context_len}h")
    print("=" * 72)

    # ── 2) 학습 데이터 전처리 (covariate 포함) ──
    # CAFI v1은 zero-shot이라 학습 데이터가 불필요하다(AutoGluon 계열만 학습 대상).
    need_training = any(k in AG_SPECS for k in model_keys)
    train_frames_target, train_frames_cov, train_names = [], [], []
    if need_training and not _all_ckpts_ready(model_keys, ckpt):
        print("\n[학습 데이터 전처리]")
        for fp in train_paths:
            res = preprocess_file(fp, include_covariates=True)
            if res is None:
                continue
            data = res["data"]
            tgt_cols = [c for c in TARGET_VARS if c in data.columns]
            if not tgt_cols:
                continue
            train_names.append(res["name"])
            train_frames_target.append(data[tgt_cols])
            train_frames_cov.append(data)  # 5변수 + covariate
        print(f"  학습 온실 {len(train_names)}개 준비")

    # ── 3) 모델 빌드/학습/로드 ──
    print("\n[모델 준비]")
    models = build_and_train_models(
        model_keys, context_len, ckpt,
        train_frames_target, train_frames_cov, train_names, ag_time_limit,
        train_paths=train_paths,
    )

    completed = _load_completed(csv_path)
    if completed:
        print(f"재개 모드: {len(completed)}개 (온실×모델×시나리오×gap×repeat) 완료됨")

    # ── 4) Test 온실 루프 ──
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

        # NMAE 정규화 상수: 온실별 변수 전체 관측 range(원래 스케일). gap/repeat/model에
        # 독립인 고정값이라 gap 길이별 비교가 공정하다(평가 지점 range로 나누면
        # gap이 커질수록 분모가 커져 NMAE가 인위적으로 작아지는 편향 발생).
        data_raw = res["data_raw"]
        norm_ranges = {
            c: float(data_raw[c].max() - data_raw[c].min())
            for c in target_cols if c in data_raw.columns
        }

        test_norm = data[target_cols]
        original_valid_mask = pd.DataFrame(
            np.where(test_norm.isna().values, 0.0, 1.0).astype(np.float32),
            index=test_norm.index, columns=test_norm.columns,
        )
        # ground_truth/마스크는 covariate까지 포함한 프레임에서 만들어 CAFI가 covariate를 쓰게 한다.
        full_norm = data[target_cols + covariate_cols]
        full_valid = pd.DataFrame(
            np.where(full_norm.isna().values, 0.0, 1.0).astype(np.float32),
            index=full_norm.index, columns=full_norm.columns,
        )
        group_labels = make_group_labels(test_norm.index)

        # 공간축 모델에 현재 타깃 온실을 알린다(이웃 캐시 갱신).
        for _m in models.values():
            if hasattr(_m, "set_greenhouse"):
                _m.set_greenhouse(greenhouse)

        print(f"\n{'='*72}\n[{greenhouse}] Test {len(test_norm)}h "
              f"| 계절: {sorted(set(group_labels['season']))} | covariate: {covariate_cols}")

        # ── base 캐시: 원본결측만 채운 전체 보간을 모델별로 1회 계산 ──
        # mask 반복마다 거대한 원본결측 gap을 재예측하던 낭비(특히 AutoGluon)를 제거한다.
        # CAFI는 covariate 포함 프레임, 나머지는 5변수 타깃 프레임 위에서 계산한다.
        bases: dict[str, object] = {}
        for mname, model in models.items():
            if not (hasattr(model, "compute_base") and hasattr(model, "impute_artificial")):
                continue
            if mname == "CAFI":
                base_in, base_mask = full_norm, full_valid
            else:
                base_in, base_mask = test_norm, original_valid_mask
            t0 = time.perf_counter()
            try:
                bases[mname] = model.compute_base(base_in, base_mask)
                print(f"    [{mname}] base 보간 캐시 ({time.perf_counter()-t0:.1f}s)")
            except Exception as e:
                print(f"    [{mname}] base 캐시 실패 → 매 mask 전체 impute: {e}")

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
                        for mname, model in models.items():
                            key = (greenhouse, mname, scenario, ",".join(mr.masked_vars),
                                   gap_h, mr.repeat)
                            if key in completed:
                                continue
                            t0 = time.perf_counter()
                            try:
                                rows = _eval_one(
                                    mname, model, mr, scalers, group_labels,
                                    greenhouse, covariate_cols, target_cols, context_len,
                                    base=bases.get(mname), norm_ranges=norm_ranges,
                                )
                            except Exception as e:
                                print(f"    [{mname}] 실패: {e}")
                                traceback.print_exc()
                                continue
                            _append_rows(csv_path, rows)
                            completed.add(key)
                            allrows = [r for r in rows if r["group_type"] == "all"]
                            mae = np.nanmean([r["MAE"] for r in allrows]) if allrows else np.nan
                            print(f"    [{mname}] {scenario}/{','.join(mr.masked_vars)} "
                                  f"gap={gap_h}h r{mr.repeat} MAE={mae:.4f} ({time.perf_counter()-t0:.1f}s)")

    # ── 5) 요약 ──
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\n{'='*72}\n비교 완료. 총 {len(df)}행 → {csv_path}")
        _print_summary(df)
        return df
    print("\n수집된 결과 없음.")
    return pd.DataFrame()


def _all_ckpts_ready(model_keys: list[str], ckpt: CheckpointManager) -> bool:
    """학습형 모델 체크포인트가 모두 존재하면 학습 데이터 전처리를 건너뛴다."""
    for k in model_keys:
        if k in AG_SPECS and not ckpt.autogluon_exists(k):
            return False
    return True


def _print_summary(df: pd.DataFrame) -> None:
    sub = df[df["group_type"] == "season"]
    if sub.empty:
        sub = df[df["group_type"] == "all"]
    print("\n[모델 × 계절 평균 NMAE]")
    try:
        tbl = sub.groupby(["model", "group_value"])["NMAE"].mean().round(4).unstack("group_value")
        print(tbl.to_string())
    except Exception as e:
        print(f"  요약 실패: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS = ["LI", "SeasonalNaive", "AG-LightGBM", "AG-RandomForest",
                  "AG-DeepAR", "AG-PatchTST", "Chronos2", "TimesFM2.5",
                  "TimesFM3.0", "CAFI", "CAFI-TimesFM3"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="온실 결측복원 10방법론 통합 비교")
    p.add_argument("--data-dir", type=str, default="../01_data")
    p.add_argument("--result-dir", type=str, default="../03_result/comparison")
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument("--scenarios", type=str, default="A,B,C")
    p.add_argument("--gap-lengths", type=str, default="6,12,24,72,168")
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--context-len", type=int, default=720)
    p.add_argument("--n-test", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ag-time-limit", type=int, default=600)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--reset-models", type=str, default="",
                   help="쉼표구분 모델명. 해당 모델 행만 results.csv에서 제거(백업 후) 후 재계산. "
                        "예: --models CAFI --reset-models CAFI")
    p.add_argument("--rebuild-split", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="스모크: Test 2온실·gap 24h·1 repeat·AG time_limit 60s")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    gap_lengths = [int(g) for g in args.gap_lengths.split(",") if g.strip()]
    n_test, n_repeats, ag_tl = args.n_test, args.n_repeats, args.ag_time_limit
    reset_models = [m.strip() for m in args.reset_models.split(",") if m.strip()]

    if args.quick:
        scenarios = ["A"]
        gap_lengths = [24]
        n_repeats = 1
        n_test = 2
        ag_tl = 60
        print("[--quick] 스모크 모드: Test 2온실 · Scenario A · gap 24h · 1 repeat · AG 60s")

    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    if not data_dir.is_absolute():
        data_dir = (_THIS_DIR / data_dir).resolve()
    if not result_dir.is_absolute():
        result_dir = (_THIS_DIR / result_dir).resolve()

    run_comparison(
        data_dir=data_dir, result_dir=result_dir, model_keys=model_keys,
        scenarios=scenarios, gap_lengths_h=gap_lengths, n_repeats=n_repeats,
        context_len=args.context_len, n_test=n_test, seed=args.seed,
        ag_time_limit=ag_tl, reset=args.reset, rebuild_split=args.rebuild_split,
        reset_models=reset_models,
    )


if __name__ == "__main__":
    main()

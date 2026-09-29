"""
foundation_model.py
파운데이션 모델 통합 인터페이스

지원 모델:
  (A) Chronos 2 (Amazon) — zero-shot / fine-tuned
  (B) TimesFM (Google) — zero-shot
  (C) PatchTST (참고용 Transformer)

공통 인터페이스: fit() / impute()
impute 전략: 결측 전후 컨텍스트를 입력으로
             Chronos-2: gap별 전체 변수를 동시 예측 (cross-variate attention)
             TimesFM / PatchTST: 변수별 독립 예측

Chronos 2 (v2):
  - Import: Chronos2Pipeline (fallback: BaseChronosPipeline)
  - predict_df()로 multivariate 예측 (group_id 공유)
  - context_len 최대 8192, prediction_length 최대 1024
  - 실제 fine-tuning 지원: pipeline.fit() → 새 파이프라인 반환
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 컨텍스트 윈도우 (14일치 기본값)
DEFAULT_CONTEXT_LEN = 336

# 대형 파운데이션 객체를 재사용해 온실 반복 실험 시 로딩 오버헤드를 줄인다.
_CHRONOS_PIPELINE_CACHE: dict[str, Any] = {}
_TIMESFM_BACKEND_CACHE: dict[tuple[str, int, int], Any] = {}


# ──────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────

class BaseImputationModel(ABC):
    name: str = "Base"

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, **kwargs) -> None: ...

    @abstractmethod
    def impute(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
    ) -> pd.DataFrame: ...

    @staticmethod
    def _find_gaps(missing: np.ndarray) -> list[tuple[int, int]]:
        """연속 결측 구간 탐지. missing[i]=True 이면 gap."""
        gaps: list[tuple[int, int]] = []
        in_gap, gap_start = False, 0
        for i, m in enumerate(missing):
            if m and not in_gap:
                gap_start, in_gap = i, True
            elif not m and in_gap:
                gaps.append((gap_start, i - 1))
                in_gap = False
        if in_gap:
            gaps.append((gap_start, len(missing) - 1))
        return gaps

    def _impute_by_segment(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
        predict_fn,           # (context_series: np.ndarray, n_pred: int) -> np.ndarray
        context_len: int = DEFAULT_CONTEXT_LEN,
        restore_future_orientation: bool = False,
    ) -> pd.DataFrame:
        """
        결측 블록을 찾아 전후 컨텍스트로 예측 후 채우는 공통 로직.
        변수별로 독립 처리.
        """
        imputed = masked_data.copy()

        for col in masked_data.columns:
            series = masked_data[col].copy()
            mask = mask_matrix[col].values  # 0=결측, 1=유효

            combined_missing = (mask == 0) | series.isna().values
            gaps = self._find_gaps(combined_missing)

            vals = series.values.copy().astype(np.float32)

            for gs, ge in gaps:
                n_pred = ge - gs + 1
                ctx_start = max(0, gs - context_len)
                context = vals[ctx_start:gs]
                context = pd.Series(context).ffill().bfill().fillna(0.0).values.astype(np.float32)
                reversed_future = False

                if len(context) == 0:
                    post_ctx = vals[ge + 1:ge + 1 + context_len]
                    post_ctx = pd.Series(post_ctx).ffill().bfill().fillna(0.0).values.astype(np.float32)
                    if len(post_ctx) == 0:
                        continue
                    context = post_ctx[::-1].copy()  # 시간 역전 (negative stride 방지)
                    reversed_future = True

                try:
                    pred = predict_fn(context, n_pred)
                    pred = np.array(pred, dtype=np.float32).flatten()
                    pred = pred[:n_pred]
                    if len(pred) < n_pred:
                        pred = np.pad(pred, (0, n_pred - len(pred)), mode="edge")
                    if restore_future_orientation and reversed_future:
                        pred = pred[::-1].copy()
                    vals[gs:ge + 1] = pred
                except Exception as e:
                    if not getattr(self, "_error_reported", False):
                        print(f"  [{self.name}] 예측 실패 (첫 발생): {e} → 해당 gap은 선형보간으로 대체")
                        self._error_reported = True

            imputed[col] = vals

        imputed = imputed.interpolate(method="linear", limit_direction="both")
        imputed = imputed.ffill().bfill()
        return imputed

    # ──────────────────────────────────────────────
    # base 캐시 기반 빠른 보간 (원본결측 1회 + 인공 gap만 재예측)
    # ──────────────────────────────────────────────
    def _impute_artificial_core(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
        artificial_bool: pd.DataFrame,   # True = 인위결측 위치
        base: pd.DataFrame,              # 원본결측이 이미 채워진 전체 보간
        batch_predict,                   # (contexts: list[np.ndarray], step:int) -> list[np.ndarray]
        horizon: int | None,
        context_len: int,
        context_index: pd.DatetimeIndex | None = None,
        restore_future_orientation: bool = False,
    ) -> pd.DataFrame:
        """
        base(원본결측까지 채운 전체 보간)에서 시작해, artificial_bool과 겹치는 gap만
        다시 예측한다. 원본결측 전용 gap은 base 값을 그대로 재사용한다.

        정확성(평가 위치 동일)의 근거:
          - 모델은 변수별 독립 예측 → 인공 gap이 없는 컬럼은 base와 동일.
          - create_gap_masks는 컬럼당 인공 gap을 1개만 만들고, 그 gap 좌측의
            원본결측 채움값은 base와 동일하므로 컨텍스트가 같다 → 재예측 결과 동일.
          - 인공 gap 우측의 순수 원본결측 채움값은 평가 대상이 아니고, 같은 컬럼에
            또 다른 인공 gap이 없으므로 base 재사용이 평가 위치에 영향을 주지 않는다.
          - gap 병합(인공+인접 원본결측)은 그대로 허용한다 — 겹치는 merged gap 전체를
            재예측하므로 기존 동작과 일치한다.
        """
        columns = list(masked_data.columns)
        arr = base[columns].values.astype(np.float32).copy()

        # ── 인공 gap과 겹치는 (merged) gap 수집 ──
        jobs: list[tuple[int, int, int, int]] = []  # (col_idx, gs, ge, n_pred)
        for ci, col in enumerate(columns):
            if col not in artificial_bool.columns:
                continue
            art = artificial_bool[col].values.astype(bool)
            if not art.any():
                continue
            mask = mask_matrix[col].values
            isna = masked_data[col].isna().values
            combined = (mask == 0) | isna
            for gs, ge in self._find_gaps(combined):
                if art[gs:ge + 1].any():
                    jobs.append((ci, gs, ge, ge - gs + 1))

        if not jobs:
            return pd.DataFrame(arr, index=masked_data.index, columns=columns)

        # ── 각 job의 컨텍스트(좌측 base 값) ──
        cur_ctx: list[np.ndarray] = []
        cur_times: list[pd.DatetimeIndex] = []
        reversed_future: list[bool] = []
        for ci, gs, ge, _ in jobs:
            ctx = arr[max(0, gs - context_len):gs, ci]
            times = context_index[max(0, gs - context_len):gs] if context_index is not None else None
            ctx = pd.Series(ctx).ffill().bfill().fillna(0.0).values.astype(np.float32)
            reversed_future.append(len(ctx) == 0)
            if len(ctx) == 0:
                post = arr[ge + 1:ge + 1 + context_len, ci]
                post = pd.Series(post).ffill().bfill().fillna(0.0).values.astype(np.float32)
                ctx = post[::-1].copy() if len(post) else ctx
                if context_index is not None:
                    times = context_index[ge + 1:ge + 1 + context_len][::-1]
            cur_ctx.append(ctx)
            if context_index is not None:
                cur_times.append(times)

        remaining = [n_pred for _, _, _, n_pred in jobs]
        collected: list[list[np.ndarray]] = [[] for _ in jobs]
        failed = [False] * len(jobs)

        # horizon=None requests each complete gap once; an integer explicitly
        # retains the legacy rolling schedule. Group only identical call spans.
        while any(remaining[j] > 0 for j in range(len(jobs))):
            groups: dict[int, list[int]] = defaultdict(list)
            for j in range(len(jobs)):
                if remaining[j] <= 0:
                    continue
                groups[remaining[j] if horizon is None else min(remaining[j], horizon)].append(j)
            for step, js in groups.items():
                ctx_list = [
                    (cur_ctx[j][-context_len:] if len(cur_ctx[j]) > context_len else cur_ctx[j])
                    for j in js
                ]
                try:
                    if context_index is None:
                        preds = batch_predict(ctx_list, step)
                    else:
                        time_list = [cur_times[j][-len(ctx):] for j, ctx in zip(js, ctx_list)]
                        preds = batch_predict(ctx_list, step, timestamps=time_list)
                except Exception as e:
                    if not getattr(self, "_art_err_reported", False):
                        print(f"  [{self.name}] artificial 예측 실패: {e} → 선형보간 대체")
                        self._art_err_reported = True
                    preds = [None] * len(js)
                for k, j in enumerate(js):
                    chunk = preds[k] if k < len(preds) else None
                    if chunk is None:
                        failed[j] = True
                        remaining[j] = 0
                        continue
                    chunk = np.asarray(chunk, dtype=np.float32).flatten()[:step]
                    collected[j].append(chunk)
                    cur_ctx[j] = np.concatenate([cur_ctx[j], chunk])
                    if context_index is not None:
                        cur_times[j] = cur_times[j].append(pd.date_range(
                            cur_times[j][-1], periods=len(chunk) + 1, freq="h")[1:])
                    remaining[j] -= step

        for j, (ci, gs, ge, n_pred) in enumerate(jobs):
            if failed[j]:
                # 예측 실패 → 해당 gap을 NaN으로 두고 마지막 선형보간에 맡긴다(기존 fallback과 동일).
                arr[gs:ge + 1, ci] = np.nan
                continue
            if not collected[j]:
                continue
            pred = np.concatenate(collected[j])[:n_pred]
            if len(pred) < n_pred:
                pred = np.pad(pred, (0, n_pred - len(pred)), mode="edge")
            if restore_future_orientation and reversed_future[j]:
                pred = pred[::-1].copy()
            arr[gs:ge + 1, ci] = pred

        out = pd.DataFrame(arr, index=masked_data.index, columns=columns)
        if any(failed):
            out = out.interpolate(method="linear", limit_direction="both").ffill().bfill()
        return out


# ──────────────────────────────────────────────
# Chronos
# ──────────────────────────────────────────────

class ChronosImputation(BaseImputationModel):
    """
    Amazon Chronos 2 기반 보간 (multivariate).
    pip install 'chronos-forecasting>=2.2'

    Multivariate 전략:
      - gap별로 모든 변수의 컨텍스트를 같은 group_id로 묶어 predict_df() 호출
      - Chronos-2의 cross-variate attention으로 변수 간 상관 활용
      - predict_df 미지원 시 변수별 predict_quantiles()로 gap 단위 fallback
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        fine_tune: bool = False,
        context_len: int = DEFAULT_CONTEXT_LEN,
        random_seed: int = 42,
        ft_num_steps: int = 500,
        ft_learning_rate: float = 1e-6,
        ft_prediction_length: int = 48,
    ):
        self.model_id = model_id
        self.fine_tune = fine_tune
        self.context_len = context_len
        self.random_seed = random_seed
        self.ft_num_steps = ft_num_steps
        self.ft_learning_rate = ft_learning_rate
        self.ft_prediction_length = ft_prediction_length
        self.pipeline: Any = None
        self.name = "Chronos2" if fine_tune else "Chronos2-zero"

        try:
            import torch
            try:
                from chronos import Chronos2Pipeline as _PipelineCls
            except ImportError:
                from chronos import BaseChronosPipeline as _PipelineCls

            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = (
                torch.bfloat16
                if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8)
                else torch.float32
            )
            cache_key = f"{model_id}|{device}"
            if cache_key not in _CHRONOS_PIPELINE_CACHE:
                _CHRONOS_PIPELINE_CACHE[cache_key] = _PipelineCls.from_pretrained(
                    model_id,
                    device_map=device,
                    dtype=torch_dtype,
                )
                print(f"  Chronos 2 로드 완료: {model_id} (device={device}, dtype={torch_dtype})")
            else:
                print(f"  Chronos 2 재사용: {model_id}")
            self.pipeline = _CHRONOS_PIPELINE_CACHE[cache_key]
        except ImportError:
            warnings.warn(
                "chronos-forecasting 미설치 또는 버전 부족.\n"
                "  pip install 'chronos-forecasting>=2.2'\n"
                "이 모델은 skip됩니다."
            )
        except Exception as e:
            warnings.warn(f"Chronos 2 로드 실패: {e}")

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        if self.pipeline is None:
            return
        if not self.fine_tune:
            print(f"  [{self.name}] zero-shot 모드 — 학습 생략")
            return

        try:
            import torch

            print(f"  [{self.name}] fine-tuning 시작 (num_steps={self.ft_num_steps})...")

            train_series = []
            for col in train_data.columns:
                vals = train_data[col].dropna().values.astype(np.float32)
                if len(vals) > self.ft_prediction_length * 2:
                    train_series.append(torch.tensor(vals, dtype=torch.float32))

            if not train_series:
                print(f"  [{self.name}] 학습 데이터 부족 → fine-tuning 생략")
                return

            # Chronos2Pipeline.fit()은 새 파이프라인을 반환한다.
            # 캐시된 원본은 수정하지 않음.
            finetuned = self.pipeline.fit(
                inputs=train_series,
                prediction_length=self.ft_prediction_length,
                finetune_mode="full",
                learning_rate=self.ft_learning_rate,
                num_steps=self.ft_num_steps,
                batch_size=min(64, len(train_series)),
            )
            self.pipeline = finetuned
            print(f"  [{self.name}] fine-tuning 완료 ({len(train_series)}개 시계열, {len(train_data)}행)")

        except AttributeError:
            warnings.warn(
                f"  [{self.name}] pipeline.fit() 미지원 → chronos-forecasting>=2.2 필요. "
                "zero-shot으로 대체합니다."
            )
        except Exception as e:
            import traceback
            warnings.warn(f"  [{self.name}] fine-tune 실패: {e}")
            traceback.print_exc()

    def impute(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.pipeline is None:
            warnings.warn(f"[{self.name}] pipeline 없음 → 선형 보간으로 대체")
            from .linear_interpolation import LinearInterpolation
            return LinearInterpolation().impute(masked_data, mask_matrix)

        return self._impute_multivariate(masked_data, mask_matrix)

    def _impute_multivariate(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        gap별로 모든 변수를 같은 group_id로 묶어 predict_df() 호출.
        Chronos-2의 cross-variate attention이 변수 간 상관을 활용한다.

        context_df 구조 (long format):
          timestamp | item_id (= gap group) | target_col | value
        """
        imputed = masked_data.copy()
        columns = list(masked_data.columns)

        any_missing = mask_matrix.eq(0).any(axis=1) | masked_data.isnull().any(axis=1)
        gap_indices = self._find_gaps(any_missing.values)

        if not gap_indices:
            return imputed.ffill().bfill()

        vals = imputed.values.copy().astype(np.float32)
        # predict_df는 timestamp 컬럼을 요구
        timestamps = pd.date_range("2020-01-01", periods=len(masked_data), freq="h")

        MIN_CTX = 3  # Chronos predict_df 최소 요구 길이

        for gs, ge in gap_indices:
            n_pred = ge - gs + 1
            ctx_start = max(0, gs - self.context_len)

            # ── context_df 구성 (wide format) ─────────────────────────
            # predict_df 입력: timestamp | item_id | col1 | col2 | ...
            # item_id는 하나("gap{gs}"), 각 변수가 별도 컬럼 → multivariate
            ctx_data: dict = {"timestamp": None}
            skip_predict_df = False

            for col_idx, col in enumerate(columns):
                ctx_vals = vals[ctx_start:gs, col_idx]
                ctx_vals = (
                    pd.Series(ctx_vals).ffill().bfill().fillna(0.0).values.astype(np.float32)
                )
                # 컨텍스트가 부족하면 gap 이후 값으로 보완 (시간 역전)
                if len(ctx_vals) < MIN_CTX:
                    post_end = min(len(vals), ge + 1 + self.context_len)
                    post_vals = vals[ge + 1:post_end, col_idx]
                    post_vals = (
                        pd.Series(post_vals).ffill().bfill().fillna(0.0).values.astype(np.float32)
                    )
                    if len(post_vals) >= MIN_CTX:
                        ctx_vals = post_vals[::-1].copy()
                    elif len(ctx_vals) + len(post_vals) >= MIN_CTX:
                        ctx_vals = np.concatenate([ctx_vals, post_vals[::-1]])
                    else:
                        skip_predict_df = True
                        break
                ctx_data[col] = ctx_vals

            if skip_predict_df:
                # 컨텍스트 부족 → fallback 경로로
                e_fallback = ValueError("컨텍스트 부족으로 predict_df 스킵")
            else:
                # 모든 컬럼의 ctx_vals 길이를 최솟값으로 맞춤
                min_len = min(len(v) for k, v in ctx_data.items() if k != "timestamp")
                for col in columns:
                    ctx_data[col] = ctx_data[col][-min_len:]
                ctx_data["timestamp"] = pd.date_range(
                    end=timestamps[gs - 1], periods=min_len, freq="h"
                )
                ctx_data["item_id"] = f"gap{gs}"
                context_df = pd.DataFrame(ctx_data)
                e_fallback = None

            if e_fallback is not None:
                use_fallback = True
            else:
                use_fallback = False
                try:
                    pred_df = self.pipeline.predict_df(
                        context_df,
                        prediction_length=n_pred,
                        quantile_levels=[0.1, 0.5, 0.9],
                        id_column="item_id",
                        timestamp_column="timestamp",
                        target=columns,   # wide-format: 변수 컬럼 리스트
                    )
                    # 반환: item_id | timestamp | target_name | predictions | 0.1 | 0.5 | 0.9
                    for col_idx, col in enumerate(columns):
                        col_pred = pred_df[pred_df["target_name"] == col].sort_values("timestamp")
                        if col_pred.empty:
                            continue
                        point = col_pred["predictions"].values[:n_pred].astype(np.float32)
                        if len(point) < n_pred:
                            point = np.pad(point, (0, n_pred - len(point)), mode="edge")
                        for i in range(n_pred):
                            if mask_matrix.iloc[gs + i, col_idx] == 0 or np.isnan(vals[gs + i, col_idx]):
                                vals[gs + i, col_idx] = point[i]
                except Exception as e:
                    if not getattr(self, "_mv_error_reported", False):
                        print(
                            f"  [{self.name}] predict_df 실패: {e}\n"
                            "  → predict_quantiles fallback으로 전환"
                        )
                        self._mv_error_reported = True
                    use_fallback = True

            if use_fallback:
                import torch
                for col_idx, col in enumerate(columns):
                    ctx_vals = vals[max(0, gs - self.context_len):gs, col_idx]
                    ctx_vals = pd.Series(ctx_vals).ffill().bfill().fillna(0.0).values.astype(np.float32)
                    if len(ctx_vals) == 0:
                        continue
                    try:
                        ctx_tensor = torch.tensor(ctx_vals.copy(), dtype=torch.float32)
                        _, mean_list = self.pipeline.predict_quantiles(
                            inputs=[ctx_tensor],
                            prediction_length=n_pred,
                            quantile_levels=[0.1, 0.5, 0.9],
                        )
                        point = mean_list[0].squeeze(0).cpu().numpy().astype(np.float32)
                        for i in range(min(n_pred, len(point))):
                            if mask_matrix.iloc[gs + i, col_idx] == 0 or np.isnan(vals[gs + i, col_idx]):
                                vals[gs + i, col_idx] = point[i]
                    except Exception:
                        pass

        for col_idx, col in enumerate(columns):
            imputed[col] = vals[:, col_idx]

        imputed = imputed.interpolate(method="linear", limit_direction="both")
        imputed = imputed.ffill().bfill()
        return imputed



# ──────────────────────────────────────────────
# TimesFM
# ──────────────────────────────────────────────

class TimesFMImputation(BaseImputationModel):
    """
    Google TimesFM 2.5 기반 보간 (zero-shot).
    pip install timesfm

    v2.5 변경사항 (vs v1.0):
      - 클래스: TimesFm(hparams=...) → TimesFM_2p5_200M_torch.from_pretrained()
      - compile(ForecastConfig(...))으로 추론 설정
      - forecast(horizon=n, inputs=[...]) → (point, quantile) 반환
      - freq 파라미터 제거
      - context_len 최대 16k, patch 배수 정렬 불필요
      - 모델 ID: timesfm-1.0-200m-pytorch → timesfm-2.5-200m-pytorch
    """

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        context_len: int = DEFAULT_CONTEXT_LEN,
        horizon_len: int = 128,
        random_seed: int = 42,
        forecasting_mode: str = "full_gap",
    ):
        self.model_id = model_id
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.random_seed = random_seed
        if forecasting_mode not in {"full_gap", "rolling"}:
            raise ValueError("forecasting_mode must be full_gap or rolling")
        self.forecasting_mode = forecasting_mode
        self.tfm: Any = None
        self.name = "TimesFM2.5"

        try:
            import timesfm
            import torch
            torch.set_float32_matmul_precision("high")

            capacity = ((max(256 if forecasting_mode == "full_gap" else horizon_len, horizon_len) + 127) // 128) * 128
            cache_key = (model_id, context_len, capacity)
            self._compiled_horizon = capacity
            if cache_key not in _TIMESFM_BACKEND_CACHE:
                tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_id)
                tfm.compile(timesfm.ForecastConfig(
                    max_context=min(context_len, 4096),
                    max_horizon=capacity,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=False,
                    fix_quantile_crossing=True,
                ))
                _TIMESFM_BACKEND_CACHE[cache_key] = tfm
                print(f"  TimesFM 2.5 로드 완료: {model_id} (context_len={context_len})")
            else:
                print(f"  TimesFM 2.5 재사용: {model_id}")
            self.tfm = _TIMESFM_BACKEND_CACHE[cache_key]
        except ImportError:
            warnings.warn("timesfm 미설치. pip install timesfm\n이 모델은 skip됩니다.")
        except Exception as e:
            import traceback
            print(f"  [TimesFM] 로드 실패: {e}")
            traceback.print_exc()

    def _ensure_horizon(self, horizon: int) -> None:
        """Grow native output capacity without changing the requested forecast span.

        The retained continuous quantile head supports at most 1024 hours.
        Longer requests fail explicitly; they are never silently chunked.
        """
        # Several adapters may share one compiled backend. Read its actual
        # capacity, rather than an instance-local value that can become stale.
        config = getattr(self.tfm, "forecast_config", None)
        actual = int(config.max_horizon) if config is not None else self._compiled_horizon
        self._compiled_horizon = actual
        if horizon <= actual:
            return
        capacity = ((int(horizon) + 127) // 128) * 128
        if capacity > 1024:
            raise ValueError("TimesFM2.5 full-gap horizon exceeds the retained 1024-h continuous quantile head")
        import timesfm
        self.tfm.compile(timesfm.ForecastConfig(
            max_context=min(self.context_len, 4096), max_horizon=capacity,
            normalize_inputs=True, use_continuous_quantile_head=True,
            force_flip_invariance=True, infer_is_positive=False,
            fix_quantile_crossing=True,
        ))
        self._compiled_horizon = capacity

    def _forecast_one(self, context: np.ndarray, horizon: int) -> np.ndarray:
        self._ensure_horizon(horizon)
        point, _ = self.tfm.forecast(horizon=horizon, inputs=[np.asarray(context, np.float32)])
        prediction = np.asarray(point[0], np.float32).flatten()[:horizon]
        if len(prediction) != horizon or not np.isfinite(prediction).all():
            raise ValueError("TimesFM2.5 returned an incomplete or nonfinite forecast")
        return prediction

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        print(f"  [{self.name}] zero-shot 모드 — 학습 생략")

    def impute(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.tfm is None:
            warnings.warn(f"[{self.name}] model 없음 → 선형 보간으로 대체")
            from .linear_interpolation import LinearInterpolation
            return LinearInterpolation().impute(masked_data, mask_matrix)

        try:
            import torch as _torch_inf
            _infer_mode = _torch_inf.inference_mode
        except ImportError:
            _infer_mode = None

        horizon = self.horizon_len
        ctx_len = self.context_len

        def predict_fn(context: np.ndarray, n_pred: int) -> np.ndarray:
            # Full-gap mode requests the whole span; rolling is explicit legacy behavior.
            results: list[np.ndarray] = []
            cur_ctx = np.asarray(context, dtype=np.float32).copy()
            remaining = n_pred

            while remaining > 0:
                chunk = remaining if self.forecasting_mode == "full_gap" else min(remaining, horizon)
                trimmed = cur_ctx[-ctx_len:] if len(cur_ctx) > ctx_len else cur_ctx
                # TimesFM forecast()는 가변 길이 입력을 받아 내부에서 pad+mask를 처리한다.
                # 여기서 edge 패딩을 강제로 넣으면 짧은 컨텍스트가 "평탄한 실제 관측"으로 오인되어
                # 초반 구간/컨텍스트 부족 구간에서 추세 추종이 약해질 수 있어 원본 길이를 그대로 전달한다.

                self._ensure_horizon(chunk)
                # v2.5: forecast(horizon=n, inputs=[array]) → (point, quantile)
                if _infer_mode is not None:
                    with _infer_mode():
                        point, _ = self.tfm.forecast(horizon=chunk, inputs=[trimmed])
                else:
                    point, _ = self.tfm.forecast(horizon=chunk, inputs=[trimmed])
                chunk_pred = np.asarray(point[0], dtype=np.float32).flatten()[:chunk]
                results.append(chunk_pred)
                cur_ctx = np.concatenate([cur_ctx, chunk_pred])[-ctx_len:]
                remaining -= chunk
            return np.concatenate(results)[:n_pred]

        return self._impute_by_segment(masked_data, mask_matrix, predict_fn, ctx_len,
                                       restore_future_orientation=self.forecasting_mode == "full_gap")

    # ── base 캐시 빠른 경로 ─────────────────────
    def compute_base(self, data: pd.DataFrame, valid_mask: pd.DataFrame) -> pd.DataFrame:
        """원본결측만 채운 전체 보간(온실당 1회 계산해 캐시)."""
        return self.impute(data, valid_mask)

    def impute_artificial(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
        artificial_bool: pd.DataFrame,
        base: pd.DataFrame,
    ) -> pd.DataFrame:
        """base에서 인공 gap만 TimesFM으로 재예측한다(원본결측은 base 재사용)."""
        if self.tfm is None:
            return base[list(masked_data.columns)].copy()

        try:
            import torch as _torch_inf
            _infer_mode = _torch_inf.inference_mode
        except ImportError:
            _infer_mode = None

        def batch_predict(contexts: list[np.ndarray], step: int) -> list[np.ndarray]:
            inputs = [np.asarray(c, dtype=np.float32) for c in contexts]
            self._ensure_horizon(step)
            if _infer_mode is not None:
                with _infer_mode():
                    point, _ = self.tfm.forecast(horizon=step, inputs=inputs)
            else:
                point, _ = self.tfm.forecast(horizon=step, inputs=inputs)
            return [np.asarray(point[k], dtype=np.float32).flatten()[:step] for k in range(len(contexts))]

        return self._impute_artificial_core(
            masked_data, mask_matrix, artificial_bool, base,
            batch_predict, None if self.forecasting_mode == "full_gap" else self.horizon_len, self.context_len,
            restore_future_orientation=self.forecasting_mode == "full_gap",
        )


# ──────────────────────────────────────────────
# TimesFM 3.0
# ──────────────────────────────────────────────

class TimesFM3Imputation(BaseImputationModel):
    """
    Google TimesFM 3.0 기반 보간 (zero-shot).
    pip install "timesfm[torch]>=3.0.1"

    v3.0 변경사항 (vs v2.5):
      - 클래스: TimesFM_2p5_200M_torch → TimesFM3Forecaster.from_pretrained(model_id, device=...)
      - compile(ForecastConfig(...)) 단계 없음. 추론 옵션은 predict() 인자로 직접 전달
      - forecast(horizon=, inputs=[...]) → predict(context=, horizon=) / predict_batch(contexts=, horizon=)
      - 반환: ForecastOutput(.forecast, .quantiles) — 2.5의 (point, quantile) 튜플과 다름
      - 아키텍처: Stacked Mixing Transformer + variate attention + iterative CPM RevIN
                  20 layers / model_dims 1280 / heads 16, 파라미터 330M (2.5는 200M)
      - 입력 패치 32, 출력 패치 64
      - 라이선스: timesfm-non-commercial-license-v1.0 (비상업 연구 목적)

    정규화·flip invariance·quantile 보정은 3.0이 내부(iterative CPM RevIN, stitching,
    linear detrending)에서 처리하므로 라이브러리 기본값을 그대로 사용한다.
    기본적으로 2.5와 같이 각 결측 구간 전체를 한 번에 요청한다.
    forecasting_mode="rolling"은 이전 128-h 분할 호출을 재현할 때만 사용한다.
    """

    def __init__(
        self,
        model_id: str = "google/timesfm-3.0-pytorch",
        context_len: int = DEFAULT_CONTEXT_LEN,
        horizon_len: int = 128,
        random_seed: int = 42,
        forecasting_mode: str = "full_gap",
    ):
        self.model_id = model_id
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.random_seed = random_seed
        if forecasting_mode not in {"full_gap", "rolling"}:
            raise ValueError("forecasting_mode must be full_gap or rolling")
        self.forecasting_mode = forecasting_mode
        self.tfm: Any = None
        self.name = "TimesFM3.0"

        try:
            import timesfm
            import torch
            torch.set_float32_matmul_precision("high")

            if not hasattr(timesfm, "TimesFM3Forecaster"):
                raise ImportError(
                    "설치된 timesfm에 TimesFM3Forecaster가 없다. "
                    "pip install 'timesfm[torch]>=3.0.1' 필요."
                )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            cache_key = (model_id, context_len, horizon_len)
            if cache_key not in _TIMESFM_BACKEND_CACHE:
                _TIMESFM_BACKEND_CACHE[cache_key] = timesfm.TimesFM3Forecaster.from_pretrained(
                    model_id, device=device,
                )
                print(f"  TimesFM 3.0 로드 완료: {model_id} "
                      f"(context_len={context_len}, device={device})")
            else:
                print(f"  TimesFM 3.0 재사용: {model_id}")
            self.tfm = _TIMESFM_BACKEND_CACHE[cache_key]
        except ImportError as e:
            warnings.warn(f"timesfm 3.0 미설치 ({e}). 이 모델은 skip됩니다.")
        except Exception as e:
            import traceback
            print(f"  [TimesFM3.0] 로드 실패: {e}")
            traceback.print_exc()

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        print(f"  [{self.name}] zero-shot 모드 — 학습 생략")

    # ── 단일 시계열 예측 ─────────────────────────
    def _forecast_one(self, context: np.ndarray, horizon: int) -> np.ndarray:
        out = self.tfm.predict(context=np.asarray(context, dtype=np.float32), horizon=horizon)
        return np.asarray(out.forecast, dtype=np.float32).flatten()[:horizon]

    def impute(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.tfm is None:
            warnings.warn(f"[{self.name}] model 없음 → 선형 보간으로 대체")
            from .linear_interpolation import LinearInterpolation
            return LinearInterpolation().impute(masked_data, mask_matrix)

        try:
            import torch as _torch_inf
            _infer_mode = _torch_inf.inference_mode
        except ImportError:
            _infer_mode = None

        horizon = self.horizon_len
        ctx_len = self.context_len

        def predict_fn(context: np.ndarray, n_pred: int) -> np.ndarray:
            # Both TimesFM versions default to one request per complete gap.
            results: list[np.ndarray] = []
            cur_ctx = np.asarray(context, dtype=np.float32).copy()
            remaining = n_pred

            while remaining > 0:
                chunk = remaining if self.forecasting_mode == "full_gap" else min(remaining, horizon)
                trimmed = cur_ctx[-ctx_len:] if len(cur_ctx) > ctx_len else cur_ctx
                # 3.0도 가변 길이 입력을 내부에서 pad+mask 처리하므로 원본 길이를 그대로 전달한다.
                if _infer_mode is not None:
                    with _infer_mode():
                        chunk_pred = self._forecast_one(trimmed, chunk)
                else:
                    chunk_pred = self._forecast_one(trimmed, chunk)
                results.append(chunk_pred)
                cur_ctx = np.concatenate([cur_ctx, chunk_pred])[-ctx_len:]
                remaining -= chunk
            return np.concatenate(results)[:n_pred]

        return self._impute_by_segment(masked_data, mask_matrix, predict_fn, ctx_len,
                                       restore_future_orientation=self.forecasting_mode == "full_gap")

    # ── base 캐시 빠른 경로 ─────────────────────
    def compute_base(self, data: pd.DataFrame, valid_mask: pd.DataFrame) -> pd.DataFrame:
        """원본결측만 채운 전체 보간(온실당 1회 계산해 캐시)."""
        return self.impute(data, valid_mask)

    def impute_artificial(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
        artificial_bool: pd.DataFrame,
        base: pd.DataFrame,
    ) -> pd.DataFrame:
        """base에서 인공 gap만 TimesFM 3.0으로 재예측한다(원본결측은 base 재사용)."""
        if self.tfm is None:
            return base[list(masked_data.columns)].copy()

        try:
            import torch as _torch_inf
            _infer_mode = _torch_inf.inference_mode
        except ImportError:
            _infer_mode = None

        def batch_predict(contexts: list[np.ndarray], step: int) -> list[np.ndarray]:
            inputs = [np.asarray(c, dtype=np.float32) for c in contexts]
            if _infer_mode is not None:
                with _infer_mode():
                    outs = list(self.tfm.predict_batch(contexts=inputs, horizon=step))
            else:
                outs = list(self.tfm.predict_batch(contexts=inputs, horizon=step))
            return [np.asarray(o.forecast, dtype=np.float32).flatten()[:step] for o in outs]

        return self._impute_artificial_core(
            masked_data, mask_matrix, artificial_bool, base,
            batch_predict, None if self.forecasting_mode == "full_gap" else self.horizon_len, self.context_len,
            restore_future_orientation=self.forecasting_mode == "full_gap",
        )


# ──────────────────────────────────────────────
# TimesFM 3.0 — 다변량 / covariate 변형 (CAFI 대비 ablation)
# ──────────────────────────────────────────────

class _TimesFM3Base(TimesFM3Imputation):
    """TimesFM 3.0 백엔드 로드를 공유하는 변형 모델의 공통 부모."""

    def _build_all_estimate(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
        estimate: pd.DataFrame | None,
    ) -> np.ndarray:
        """
        모든 변수의 '현재 최선 추정치' (n, F) 행렬.

        관측 구간은 실측값, 결측 구간(원본결측 + 인공 gap)은 `estimate`(모델 자신의
        단변량 Round-0 보간) 값을 쓴다. CAFI의 merged_all과 같은 역할이며 반복은 없다.

        주의 — 여기에 run_comparison의 `base`(원본결측만 채운 배열)를 넣으면 안 된다.
        base는 인공 gap 위치에 정답값을 그대로 갖고 있어 covariate로 쓰는 순간
        누수가 된다. 반드시 마스킹을 반영한 Round-0 결과를 넘겨야 한다.
        """
        cols = list(masked_data.columns)
        if estimate is not None:
            arr = estimate[cols].values.astype(np.float32).copy()
        else:
            arr = (masked_data[cols]
                   .interpolate(method="linear", limit_direction="both")
                   .ffill().bfill().values.astype(np.float32))
        obs = masked_data[cols].values.astype(np.float32)
        valid = (mask_matrix[cols].values != 0) & ~np.isnan(obs)
        arr[valid] = obs[valid]  # 관측된 곳은 실측값 우선
        return arr


class TimesFM3MVImputation(_TimesFM3Base):
    """
    TimesFM 3.0 다변량(variate attention) 보간.

    predict(context=2D (n_variates, ctx_len))로 5개 변수를 **동시에** 예측한다.
    변수 간 정보는 오직 '과거 구간'을 통해서만 흐른다 — gap 구간에서는 어떤 변수도
    관측값으로 주어지지 않으므로, CAFI처럼 'gap 내 다른 변수의 실측값을 조건으로
    쓰는' 메커니즘은 아니다. 이 차이를 분리하기 위한 arm이다.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "TimesFM3.0-MV"

    def _predict_target(
        self,
        all_est: np.ndarray,
        col_idx: int,
        gs: int,
        ge: int,
        index: pd.Index,
    ) -> np.ndarray:
        n_pred = ge - gs + 1
        ctx_start = max(0, gs - self.context_len)
        ctx2d = all_est[ctx_start:gs, :].T.astype(np.float32)  # (F, ctx_len)
        if ctx2d.shape[1] == 0:
            raise ValueError("empty context")

        results: list[np.ndarray] = []
        cur = ctx2d.copy()
        remaining = n_pred
        while remaining > 0:
            chunk = remaining if self.forecasting_mode == "full_gap" else min(remaining, self.horizon_len)
            trimmed = cur[:, -self.context_len:] if cur.shape[1] > self.context_len else cur
            out = self.tfm.predict(context=trimmed, horizon=chunk)
            fc = np.asarray(out.forecast, dtype=np.float32)   # (F, chunk)
            fc = fc.reshape(ctx2d.shape[0], -1)[:, :chunk]
            results.append(fc[col_idx].copy())
            cur = np.concatenate([cur, fc], axis=1)[:, -self.context_len:]
            remaining -= chunk
        return np.concatenate(results)[:n_pred]

    def impute(self, masked_data: pd.DataFrame, mask_matrix: pd.DataFrame) -> pd.DataFrame:
        return self._impute_variant(masked_data, mask_matrix, estimate=None)

    def compute_base(self, data: pd.DataFrame, valid_mask: pd.DataFrame) -> pd.DataFrame:
        # 다른 변수의 결측 구간을 채울 단변량 초기 추정치는 부모(2.5와 동일 전략)를 쓴다.
        return TimesFM3Imputation.impute(self, data, valid_mask)

    def _round0(self, masked_data, mask_matrix, artificial_bool, base):
        """마스킹을 반영한 단변량 Round-0 추정치(= CAFI Round 0)."""
        return TimesFM3Imputation.impute_artificial(
            self, masked_data, mask_matrix, artificial_bool, base)

    def impute_artificial(self, masked_data, mask_matrix, artificial_bool, base):
        # MV는 gap **이전** 구간만 컨텍스트로 읽으므로 base를 그대로 써도 누수가 없다
        # (인공 gap 위치의 base 값은 참조하지 않는다).
        return self._impute_variant(masked_data, mask_matrix, estimate=base,
                                    artificial_bool=artificial_bool)

    def _impute_variant(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
        estimate: pd.DataFrame | None,
        artificial_bool: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self.tfm is None:
            from .linear_interpolation import LinearInterpolation
            return LinearInterpolation().impute(masked_data, mask_matrix)

        try:
            import torch as _t
            _infer = _t.inference_mode
        except ImportError:
            _infer = None

        cols = list(masked_data.columns)
        all_est = self._build_all_estimate(masked_data, mask_matrix, estimate)
        out = (estimate[cols].values.astype(np.float32).copy() if estimate is not None
               else all_est.copy())

        for ci, col in enumerate(cols):
            combined = (mask_matrix[col].values == 0) | masked_data[col].isna().values
            for gs, ge in self._find_gaps(combined):
                if artificial_bool is not None:
                    if col not in artificial_bool.columns:
                        continue
                    if not artificial_bool[col].values[gs:ge + 1].any():
                        continue  # 인공 gap이 없는 구간은 base 재사용
                if gs == 0:
                    # A natural leading gap may join an artificial interval.
                    # No past context exists: retain the masked-input estimate.
                    # This is an availability fallback, not a backend failure.
                    continue
                try:
                    if _infer is not None:
                        with _infer():
                            pred = self._predict_target(all_est, ci, gs, ge, masked_data.index)
                    else:
                        pred = self._predict_target(all_est, ci, gs, ge, masked_data.index)
                    out[gs:ge + 1, ci] = pred[:ge - gs + 1]
                except Exception as e:
                    if not getattr(self, "_error_reported", False):
                        print(f"  [{self.name}] 예측 실패 (첫 발생): {e}")
                        self._error_reported = True

        res = pd.DataFrame(out, index=masked_data.index, columns=cols)
        return res.interpolate(method="linear", limit_direction="both").ffill().bfill()


class TimesFM3CovImputation(TimesFM3MVImputation):
    """
    TimesFM 3.0 covariate 보간 (past_future_covariates) — 'CAFI 1라운드' 등가물.

    target은 단변량으로 두고, 나머지 4개 변수 + 캘린더 feature를
    past_future_covariates(과거 ctx_len + gap 구간 n_pred)로 제공한다.
    gap 구간의 다른 변수 값은 **관측된 경우에만 실측값**을 쓰고, 마스킹된 경우에는
    자기 자신의 단변량 Round-0 추정치를 쓴다(정답값을 절대 참조하지 않는다).
    이는 CAFI의 covariate 구성과 동일하며, 차이는 **반복 refinement 없음(1-pass)** 이다.

    따라서 (CAFI − 이 모델)의 차이가 곧 'iterative refinement의 순수 기여'다.

    시나리오별 covariate 가용성:
      A: 나머지 4변수 전부 gap 내 관측 → 실측 covariate
      B: 일부만 관측 → 실측 + Round-0 혼합
      C: 전부 결측 → Round-0 추정치만 (CAFI Round 1과 동일 조건)
    """

    def __init__(self, use_time_covariates: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_time_covariates = use_time_covariates
        self.name = "TimesFM3.0-COV"

    def impute(self, masked_data: pd.DataFrame, mask_matrix: pd.DataFrame) -> pd.DataFrame:
        # Round 0: 마스킹을 반영한 단변량 보간 → covariate 소스
        round0 = TimesFM3Imputation.impute(self, masked_data, mask_matrix)
        return self._impute_variant(masked_data, mask_matrix, estimate=round0)

    def impute_artificial(self, masked_data, mask_matrix, artificial_bool, base):
        # base를 covariate로 쓰면 인공 gap 위치의 정답이 새어 들어간다.
        # 반드시 마스킹이 반영된 Round-0 결과를 covariate 소스로 넘긴다.
        round0 = self._round0(masked_data, mask_matrix, artificial_bool, base)
        return self._impute_variant(masked_data, mask_matrix, estimate=round0,
                                    artificial_bool=artificial_bool)

    @staticmethod
    def _time_cov(index: pd.Index) -> np.ndarray:
        dt = pd.DatetimeIndex(index)
        hour = dt.hour.to_numpy(dtype=np.float32)
        doy = dt.dayofyear.to_numpy(dtype=np.float32)
        return np.stack([
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
        ]).astype(np.float32)

    def _predict_target(
        self,
        all_est: np.ndarray,
        col_idx: int,
        gs: int,
        ge: int,
        index: pd.Index,
    ) -> np.ndarray:
        n_pred = ge - gs + 1
        ctx_start = max(0, gs - self.context_len)
        target_ctx = all_est[ctx_start:gs, col_idx].astype(np.float32)
        if len(target_ctx) == 0:
            raise ValueError("empty context")

        other = [j for j in range(all_est.shape[1]) if j != col_idx]

        results: list[np.ndarray] = []
        cur_ctx = target_ctx.copy()
        pos = gs
        remaining = n_pred
        while remaining > 0:
            chunk = remaining if self.forecasting_mode == "full_gap" else min(remaining, self.horizon_len)
            t_ctx = cur_ctx[-self.context_len:] if len(cur_ctx) > self.context_len else cur_ctx
            c0 = pos - len(t_ctx)
            # covariate: 과거(t_ctx 구간) + 미래(gap chunk 구간)
            cov_past = all_est[c0:pos, :][:, other].T                   # (n_other, len(t_ctx))
            cov_fut = all_est[pos:pos + chunk, :][:, other].T           # (n_other, chunk)
            pf = np.concatenate([cov_past, cov_fut], axis=1).astype(np.float32)
            if self.use_time_covariates:
                tc = self._time_cov(index[c0:pos + chunk])
                pf = np.concatenate([pf, tc], axis=0)
            if pf.shape[1] != len(t_ctx) + chunk:
                raise ValueError(f"covariate 길이 불일치: {pf.shape[1]} != {len(t_ctx) + chunk}")

            out = self.tfm.predict(
                context=t_ctx, horizon=chunk,
                past_future_covariates=pf, padding_mode="edge",
            )
            pred = np.asarray(out.forecast, dtype=np.float32).flatten()[:chunk]
            results.append(pred)
            cur_ctx = np.concatenate([cur_ctx, pred])[-self.context_len:]
            pos += chunk
            remaining -= chunk
        return np.concatenate(results)[:n_pred]


# ──────────────────────────────────────────────
# CAFI — Covariate-Aware Foundation Imputation
# ──────────────────────────────────────────────

class CAFIImputation(BaseImputationModel):
    """
    CAFI: Covariate-Aware Foundation Imputation

    핵심 아이디어:
      기존 Chronos 2는 변수별 독립 예측 → 변수 간 물리적 상관관계 미활용.
      CAFI는 Chronos 2의 공식 covariate API를 활용해 imputation 시
      나머지 변수들을 past/future covariate로 제공, 수렴 기반 iterative refinement 수행.

    알고리즘:
      Round 0: 기존 Chronos 2 방식으로 초기 imputation (covariate 없음)
      Round k (k≥1):
        for each variable X:
          past_covariates  = gap 이전 구간의 나머지 변수 관측값
          future_covariates = gap 구간의 나머지 변수 값
            - individual 마스킹: 실제 관측값 사용 (해당 변수가 관측된 경우)
            - block 마스킹: 이전 Round 결과 사용
          X_imputed = Chronos2.predict(target=X, covariates=others)
        수렴 판정: RMSE 변화율 < tol 또는 max_rounds 도달 시 종료

    논문 비교 라인업:
      LI | LSTM | PatchTST | TimesFM2.5 | Chronos2 | CAFI
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        fine_tune: bool = False,
        context_len: int = DEFAULT_CONTEXT_LEN,
        random_seed: int = 42,
        max_rounds: int = 5,
        tol: float = 1e-3,
        var_tol: float = 5e-4,
        ft_num_steps: int = 500,
        ft_learning_rate: float = 1e-6,
        ft_prediction_length: int = 48,
        use_std_normalization: bool = True,
        use_time_covariates: bool = True,
        name: str | None = None,
    ):
        self.model_id = model_id
        self.fine_tune = fine_tune
        self.context_len = context_len
        self.random_seed = random_seed
        self.max_rounds = max_rounds
        self.tol = tol
        self.var_tol = var_tol
        self.ft_num_steps = ft_num_steps
        self.ft_learning_rate = ft_learning_rate
        self.ft_prediction_length = ft_prediction_length
        self.use_std_normalization = use_std_normalization
        self.use_time_covariates = use_time_covariates
        self.pipeline: Any = None
        self.name = name or ("CAFI-Chronos" if fine_tune else "CAFI-Chronos-zero")
        # CAFI backbone과 주요 추론 설정을 캐시에 포함해 서로 다른 알고리즘 결과가 섞이지 않게 한다.
        self.cache_signature = (
            f"chronos_cafi_rounds{max_rounds}_tol{tol:g}_vartol{var_tol:g}_"
            f"std{int(use_std_normalization)}_time{int(use_time_covariates)}"
        )
        self._scale_mean: pd.Series | None = None
        self._scale_std: pd.Series | None = None

        try:
            import torch
            try:
                from chronos import Chronos2Pipeline as _PipelineCls
            except ImportError:
                from chronos import BaseChronosPipeline as _PipelineCls

            device = "cuda" if torch.cuda.is_available() else "cpu"
            torch_dtype = (
                torch.bfloat16
                if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8)
                else torch.float32
            )
            # Chronos 2 파이프라인은 ChronosImputation과 공유 (캐시 재사용)
            cache_key = f"{model_id}|{device}"
            if cache_key not in _CHRONOS_PIPELINE_CACHE:
                _CHRONOS_PIPELINE_CACHE[cache_key] = _PipelineCls.from_pretrained(
                    model_id,
                    device_map=device,
                    dtype=torch_dtype,
                )
                print(f"  CAFI 로드 완료: {model_id} (device={device})")
            else:
                print(f"  CAFI 재사용: {model_id}")
            self.pipeline = _CHRONOS_PIPELINE_CACHE[cache_key]
        except ImportError:
            warnings.warn(
                "chronos-forecasting 미설치.\n"
                "  pip install 'chronos-forecasting>=2.2'\n"
                "이 모델은 skip됩니다."
            )
        except Exception as e:
            warnings.warn(f"CAFI 로드 실패: {e}")

    # ── fit ────────────────────────────────────

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        # 변수별 표준편차 차이를 줄이기 위해 CAFI 입력 스케일을 train 통계로 맞춘다.
        self._fit_standardizer(train_data)
        if self.pipeline is None:
            return
        if not self.fine_tune:
            print(
                f"  [{self.name}] zero-shot 모드 — 학습 생략 "
                f"(context_len={self.context_len}, std_norm={self.use_std_normalization}, "
                f"time_covariates={self.use_time_covariates})"
            )
            return

        try:
            import torch
            print(f"  [{self.name}] fine-tuning 시작 (num_steps={self.ft_num_steps})...")
            train_scaled = self._transform_frame(train_data)
            train_series = []
            for col in train_scaled.columns:
                vals = train_scaled[col].dropna().values.astype(np.float32)
                if len(vals) > self.ft_prediction_length * 2:
                    train_series.append(torch.tensor(vals, dtype=torch.float32))
            if not train_series:
                print(f"  [{self.name}] 학습 데이터 부족 → fine-tuning 생략")
                return
            finetuned = self.pipeline.fit(
                inputs=train_series,
                prediction_length=self.ft_prediction_length,
                finetune_mode="full",
                learning_rate=self.ft_learning_rate,
                num_steps=self.ft_num_steps,
                batch_size=min(64, len(train_series)),
            )
            self.pipeline = finetuned
            print(f"  [{self.name}] fine-tuning 완료 ({len(train_series)}개 시계열)")
        except AttributeError:
            warnings.warn(f"  [{self.name}] pipeline.fit() 미지원 → zero-shot으로 대체")
        except Exception as e:
            import traceback
            warnings.warn(f"  [{self.name}] fine-tune 실패: {e}")
            traceback.print_exc()

    # ── impute ─────────────────────────────────

    def impute(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        import time as _time

        if self.pipeline is None:
            warnings.warn(f"[{self.name}] pipeline 없음 → 선형 보간으로 대체")
            from .linear_interpolation import LinearInterpolation
            return LinearInterpolation().impute(masked_data, mask_matrix)

        columns = list(masked_data.columns)
        n_vars = len(columns)
        t_total_start = _time.perf_counter()
        work_data = self._transform_frame(masked_data)

        # masked_data 보간값은 수렴 판정용으로 매 round 재사용 → 1회만 계산
        _masked_interp = (
            work_data.copy()
            .interpolate(method="linear", limit_direction="both")
            .ffill().bfill().fillna(0.0)
            .values.astype(np.float32)
        )

        # ── Round 0: 초기 imputation (covariate 없음, 변수별 독립) ──
        t0 = _time.perf_counter()
        current = self._round0_impute(work_data, mask_matrix, columns)
        prev_rmse = self._compute_pseudo_rmse(current, _masked_interp, mask_matrix)
        t0_elapsed = _time.perf_counter() - t0

        print(f"  [{self.name}] Round 0 done  "
              f"(pseudo-RMSE={prev_rmse:.6f}, elapsed={t0_elapsed:.1f}s, "
              f"context_len={self.context_len})")

        # ── Round k: Covariate-aware iterative refinement ──
        prev_refined = None
        for rnd in range(1, self.max_rounds + 1):
            tk = _time.perf_counter()
            refined = self._covariate_round(
                current=current,
                masked_data=work_data,
                mask_matrix=mask_matrix,
                columns=columns,
                prev_refined=prev_refined,
            )
            curr_rmse = self._compute_pseudo_rmse(refined, _masked_interp, mask_matrix)
            change = abs(curr_rmse - prev_rmse) / (prev_rmse + 1e-12)
            tk_elapsed = _time.perf_counter() - tk

            print(f"  [{self.name}] Round {rnd} done  "
                  f"(pseudo-RMSE={curr_rmse:.6f}, Δ={change:.4f}, "
                  f"elapsed={tk_elapsed:.1f}s)")

            prev_refined = current
            current = refined
            if change < self.tol:
                print(f"  [{self.name}] Converged (Δ={change:.4f} < tol={self.tol})")
                break
            prev_rmse = curr_rmse

        t_total = _time.perf_counter() - t_total_start
        final_rounds = rnd if rnd <= self.max_rounds else self.max_rounds
        print(f"  [{self.name}] Total impute time: {t_total:.1f}s "
              f"({final_rounds} rounds)")

        return self._inverse_transform_frame(current)

    # ── 내부 메서드 ────────────────────────────

    def _round0_impute(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        """Round 0: covariate 없이 변수별 독립 예측 (기존 Chronos2 방식)."""
        import torch

        imputed = masked_data.copy()
        mask_vals = mask_matrix.values.astype(np.float32)

        for col_idx, col in enumerate(columns):
            col_mask = mask_vals[:, col_idx]
            series = masked_data[col].values.astype(np.float32)

            gaps = self._find_gaps(col_mask == 0)
            for gs, ge in gaps:
                n_pred = ge - gs + 1
                ctx_start = max(0, gs - self.context_len)
                ctx = series[ctx_start:gs]
                ctx = pd.Series(ctx).ffill().bfill().fillna(0.0).values.astype(np.float32)
                if len(ctx) == 0:
                    post = series[ge + 1:ge + 1 + self.context_len]
                    post = pd.Series(post).ffill().bfill().fillna(0.0).values.astype(np.float32)
                    if len(post) == 0:
                        continue
                    ctx = post[::-1].copy()

                try:
                    point = self._predict_univariate(ctx, n_pred)
                    imputed.iloc[gs:ge + 1, col_idx] = point[:n_pred]
                except Exception as e:
                    if not getattr(self, "_r0_err_reported", False):
                        print(f"  [{self.name}] Round0 예측 실패: {e}")
                        self._r0_err_reported = True

        imputed = imputed.interpolate(method="linear", limit_direction="both").ffill().bfill()
        return imputed

    def _covariate_round(
        self,
        current: pd.DataFrame,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
        columns: list[str],
        prev_refined: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Round k: 각 변수를 나머지 변수를 covariate로 사용해 재예측.

        Optimizations:
          - per-variable early exit: 이전 round 대비 변수별 RMSE 변화 < var_tol 이면 skip
          - vectorized covariate construction: numpy 배열 연산으로 텐서 생성
        """
        import time as _time
        import torch

        refined = current.copy()
        mask_vals = mask_matrix.values.astype(np.float32)
        orig_vals = masked_data.values.astype(np.float32)
        # 관측값 우선 병합 행렬: 전체 (T, F) 한번에 미리 계산
        cur_vals = current.values.astype(np.float32)
        merged_all = np.where(mask_vals == 1, orig_vals, cur_vals).astype(np.float32)

        for col_idx, col in enumerate(columns):
            _t_var = _time.perf_counter()
            col_mask = mask_vals[:, col_idx]
            gaps = self._find_gaps(col_mask == 0)
            if not gaps:
                continue

            # ── per-variable early exit ──
            if prev_refined is not None:
                prev_col = prev_refined[col].values.astype(np.float32)
                curr_col = current[col].values.astype(np.float32)
                missing = col_mask == 0
                if missing.sum() > 0:
                    var_change = float(np.sqrt(np.mean((curr_col[missing] - prev_col[missing]) ** 2)))
                    if var_change < self.var_tol:
                        continue

            other_idxs = [columns.index(c) for c in columns if c != col]
            other_cols = [c for c in columns if c != col]
            _n_gaps = len(gaps)
            _n_infer = 0

            for gs, ge in gaps:
                n_pred = ge - gs + 1
                ctx_start = max(0, gs - self.context_len)

                # ── target context ──
                target_ctx = cur_vals[ctx_start:gs, col_idx].copy()
                _nan_mask = np.isnan(target_ctx)
                if _nan_mask.any():
                    s = pd.Series(target_ctx)
                    target_ctx = s.ffill().bfill().fillna(0.0).values.astype(np.float32)
                if len(target_ctx) == 0:
                    continue

                # ── covariates: vectorized construction ──
                # past_covariates: (ctx_len, n_other) 슬라이스 → per-col tensor
                past_slice = merged_all[ctx_start:gs]  # (ctx_len, F)
                past_covs: dict[str, Any] = {}
                for oc, oi in zip(other_cols, other_idxs):
                    v = past_slice[:, oi].copy()
                    _nm = np.isnan(v)
                    if _nm.any():
                        v = pd.Series(v).ffill().bfill().fillna(0.0).values.astype(np.float32)
                    past_covs[oc] = torch.tensor(v, dtype=torch.float32)

                if self.use_time_covariates:
                    # 시간 정보는 미래 값을 보지 않는 calendar feature라 block 결측에서도 사용할 수 있다.
                    past_time = self._time_covariates(masked_data.index[ctx_start:gs])
                    for tc in past_time.columns:
                        past_covs[tc] = torch.tensor(past_time[tc].values, dtype=torch.float32)

                # future_covariates: gap 구간
                future_slice = merged_all[gs:ge + 1]  # (n_pred, F)
                future_covs: dict[str, Any] = {}
                for oc, oi in zip(other_cols, other_idxs):
                    v = future_slice[:, oi].copy()
                    _nm = np.isnan(v)
                    if _nm.any():
                        v = pd.Series(v).ffill().bfill().fillna(0.0).values.astype(np.float32)
                    future_covs[oc] = torch.tensor(v, dtype=torch.float32)

                if self.use_time_covariates:
                    # gap 구간의 시간 feature를 future covariate로 제공해 일주기·계절 위치를 알려준다.
                    future_time = self._time_covariates(masked_data.index[gs:ge + 1])
                    for tc in future_time.columns:
                        future_covs[tc] = torch.tensor(future_time[tc].values, dtype=torch.float32)

                # ── 백본 predict() with covariate (백본별 구현) ──
                try:
                    _n_infer += 1
                    point = self._predict_with_covariates(
                        target_ctx, past_covs, future_covs, n_pred,
                    )
                    point = point[:n_pred]
                    refined.iloc[gs:ge + 1, col_idx] = point

                except Exception as e:
                    if not getattr(self, "_cov_err_reported", False):
                        print(f"  [{self.name}] covariate predict failed: {e} → keep prev round")
                        self._cov_err_reported = True

            _t_var_elapsed = _time.perf_counter() - _t_var
            print(f"    [{self.name}] var={col}  gaps={_n_gaps}  "
                  f"infer_calls={_n_infer}  elapsed={_t_var_elapsed:.2f}s")

        refined = refined.interpolate(method="linear", limit_direction="both").ffill().bfill()
        return refined

    def _predict_univariate(self, ctx: np.ndarray, n_pred: int) -> np.ndarray:
        """Round 0용 covariate 없는 단변량 예측 (백본별 구현). Chronos-2 기본 구현."""
        import torch
        ctx_tensor = torch.tensor(np.asarray(ctx, dtype=np.float32), dtype=torch.float32)
        _, mean_list = self.pipeline.predict_quantiles(
            inputs=[ctx_tensor],
            prediction_length=n_pred,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        return mean_list[0].squeeze(0).cpu().numpy().astype(np.float32)

    def _predict_with_covariates(
        self,
        target_ctx: np.ndarray,
        past_covs: dict,
        future_covs: dict,
        n_pred: int,
    ) -> np.ndarray:
        """
        covariate를 조건으로 target의 gap 구간을 예측한다 (백본별 구현).

        Chronos-2 구현: predict(inputs=[{target, past_covariates, future_covariates}]).
        past_covs / future_covs는 {컬럼명: 1D 텐서} 형태이며 키 집합이 서로 일치한다.
        """
        import torch
        input_dict = {
            "target": torch.tensor(target_ctx, dtype=torch.float32),
            "past_covariates": past_covs,
            "future_covariates": future_covs,
        }
        forecasts = self.pipeline.predict(inputs=[input_dict], prediction_length=n_pred)
        # forecasts[0]: (n_variates, n_quantiles, pred_len)
        pred_tensor = forecasts[0]
        if pred_tensor.ndim == 3:
            n_q = pred_tensor.shape[1]
            median_idx = n_q // 2
            return pred_tensor[0, median_idx, :].cpu().numpy().astype(np.float32)
        return pred_tensor.cpu().numpy().flatten().astype(np.float32)

    def _fit_standardizer(self, train_data: pd.DataFrame) -> None:
        # train 통계만 사용해 변수별 평균과 표준편차를 저장한다.
        filled = train_data.copy().interpolate(method="linear", limit_direction="both").ffill().bfill()
        self._scale_mean = filled.mean(axis=0)
        self._scale_std = filled.std(axis=0).replace(0, 1.0).fillna(1.0)

    def _transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        # CAFI 내부 입력을 변수별 z-score로 바꿔 큰 분산 변수가 예측을 지배하지 않게 한다.
        if not self.use_std_normalization or self._scale_mean is None or self._scale_std is None:
            return df.copy()
        return (df - self._scale_mean.reindex(df.columns)) / self._scale_std.reindex(df.columns)

    def _inverse_transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        # CAFI 출력은 다시 기존 평가 파이프라인이 기대하는 원래 정규화 스케일로 되돌린다.
        if not self.use_std_normalization or self._scale_mean is None or self._scale_std is None:
            return df.copy()
        return df * self._scale_std.reindex(df.columns) + self._scale_mean.reindex(df.columns)

    @staticmethod
    def _time_covariates(index: pd.Index) -> pd.DataFrame:
        # hour/day-of-year의 sin/cos 표현으로 주기적인 시간 정보를 연속 feature로 제공한다.
        dt_index = pd.DatetimeIndex(index)
        hour = dt_index.hour.to_numpy(dtype=np.float32)
        dayofyear = dt_index.dayofyear.to_numpy(dtype=np.float32)
        return pd.DataFrame({
            "time_hour_sin": np.sin(2 * np.pi * hour / 24.0).astype(np.float32),
            "time_hour_cos": np.cos(2 * np.pi * hour / 24.0).astype(np.float32),
            "time_doy_sin": np.sin(2 * np.pi * dayofyear / 365.25).astype(np.float32),
            "time_doy_cos": np.cos(2 * np.pi * dayofyear / 365.25).astype(np.float32),
        }, index=dt_index)

    @staticmethod
    def _compute_pseudo_rmse(
        imputed: pd.DataFrame,
        masked_interp: np.ndarray,
        mask_matrix: pd.DataFrame,
    ) -> float:
        """
        수렴 판정용 pseudo-RMSE.
        결측 위치(mask=0)에서 imputed와 masked_interp(선형보간 초기값, 사전 계산)의 차이.
        실제 ground truth 없이 round 간 변화량만 측정.
        """
        mask_vals = mask_matrix.values.astype(np.float32)
        imp_vals = imputed.values.astype(np.float32)
        missing = mask_vals == 0
        if missing.sum() == 0:
            return 0.0
        diff = imp_vals[missing] - masked_interp[missing]
        return float(np.sqrt(np.mean(diff ** 2)))


class CAFITimesFM3Imputation(CAFIImputation):
    """
    CAFI의 백본을 Chronos-2 → TimesFM 3.0으로 교체한 변형.

    알고리즘(Round 0 → covariate refinement → 수렴 판정)은 CAFIImputation과 **완전히 동일**하며,
    바뀌는 것은 covariate 조건부 예측을 수행하는 백본뿐이다:
      - Chronos-2 : predict(inputs=[{target, past_covariates, future_covariates}])
      - TimesFM3.0: predict(context=target, past_future_covariates=(n_cov, ctx+horizon))

    CAFI의 past/future covariate dict를 TimesFM 3.0이 요구하는 2D 배열로 변환한다.
    두 dict의 키 집합은 동일하므로(타 변수 + 캘린더 feature) 정렬된 공통 키 순서로 쌓는다.
    """

    def __init__(
        self,
        model_id: str = "google/timesfm-3.0-pytorch",
        context_len: int = DEFAULT_CONTEXT_LEN,
        name: str | None = None,
        **kwargs,
    ):
        # 부모의 Chronos 로딩을 건너뛰고 TimesFM 3.0 백엔드를 직접 준비한다.
        kwargs.pop("fine_tune", None)
        self.model_id = model_id
        self.context_len = context_len
        self.fine_tune = False
        self.random_seed = kwargs.pop("random_seed", 42)
        self.max_rounds = kwargs.pop("max_rounds", 5)
        self.tol = kwargs.pop("tol", 1e-3)
        self.var_tol = kwargs.pop("var_tol", 5e-4)
        self.use_std_normalization = kwargs.pop("use_std_normalization", True)
        self.use_time_covariates = kwargs.pop("use_time_covariates", True)
        # 부모 코드가 참조하는 fine-tune 관련 속성(미사용)도 채워둔다.
        self.ft_num_steps = kwargs.pop("ft_num_steps", 500)
        self.ft_learning_rate = kwargs.pop("ft_learning_rate", 1e-6)
        self.ft_prediction_length = kwargs.pop("ft_prediction_length", 48)
        self.pipeline: Any = None
        self.tfm: Any = None
        self.name = name or "CAFI-TimesFM3"
        self.cache_signature = (
            f"tfm3_cafi_rounds{self.max_rounds}_tol{self.tol:g}_"
            f"vartol{self.var_tol:g}_std{int(self.use_std_normalization)}_"
            f"time{int(self.use_time_covariates)}"
        )
        self._scale_mean: pd.Series | None = None
        self._scale_std: pd.Series | None = None
        self.uses_ext_covariates = False

        try:
            import timesfm
            import torch
            torch.set_float32_matmul_precision("high")
            if not hasattr(timesfm, "TimesFM3Forecaster"):
                raise ImportError("timesfm>=3.0.1 필요 (TimesFM3Forecaster 없음)")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cache_key = (model_id, "cafi", context_len)
            if cache_key not in _TIMESFM_BACKEND_CACHE:
                _TIMESFM_BACKEND_CACHE[cache_key] = timesfm.TimesFM3Forecaster.from_pretrained(
                    model_id, device=device,
                )
                print(f"  CAFI-TimesFM3 로드 완료: {model_id} (device={device})")
            else:
                print(f"  CAFI-TimesFM3 재사용: {model_id}")
            self.tfm = _TIMESFM_BACKEND_CACHE[cache_key]
            self.pipeline = self.tfm  # 부모의 `pipeline is None` 가드를 통과시킨다
        except ImportError as e:
            warnings.warn(f"timesfm 3.0 미설치 ({e}). 이 모델은 skip됩니다.")
        except Exception as e:
            warnings.warn(f"CAFI-TimesFM3 로드 실패: {e}")

    def _predict_univariate(self, ctx: np.ndarray, n_pred: int) -> np.ndarray:
        """Round 0: TimesFM 3.0 단변량 예측 (긴 horizon도 네이티브 지원)."""
        out = self.tfm.predict(
            context=np.asarray(ctx, dtype=np.float32).flatten(), horizon=n_pred,
        )
        return np.asarray(out.forecast, dtype=np.float32).flatten()[:n_pred]

    def _predict_with_covariates(
        self,
        target_ctx: np.ndarray,
        past_covs: dict,
        future_covs: dict,
        n_pred: int,
    ) -> np.ndarray:
        """TimesFM 3.0의 past_future_covariates 인터페이스로 동일 조건 예측을 수행한다."""
        ctx = np.asarray(target_ctx, dtype=np.float32).flatten()
        keys = [k for k in past_covs.keys() if k in future_covs]
        rows = []
        for k in keys:
            pv = np.asarray(past_covs[k], dtype=np.float32).flatten()
            fv = np.asarray(future_covs[k], dtype=np.float32).flatten()[:n_pred]
            if len(pv) != len(ctx) or len(fv) != n_pred:
                raise ValueError(
                    f"covariate '{k}' 길이 불일치: past {len(pv)} vs ctx {len(ctx)}, "
                    f"future {len(fv)} vs n_pred {n_pred}"
                )
            rows.append(np.concatenate([pv, fv]))
        pf = np.stack(rows).astype(np.float32) if rows else None

        out = self.tfm.predict(
            context=ctx, horizon=n_pred,
            past_future_covariates=pf, padding_mode="edge",
        )
        return np.asarray(out.forecast, dtype=np.float32).flatten()[:n_pred]


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def build_foundation_models(
    include: list[str] | None = None,
    context_len: int = DEFAULT_CONTEXT_LEN,
) -> list[BaseImputationModel]:
    """
    요청된 파운데이션 모델 리스트 생성.

    include 예시: ['chronos_ft', 'timesfm_zero', 'timesfm3_zero', 'cafi_chronos']
    None이면 전체 시도.

    Chronos 키는 v1과 동일(chronos_zero, chronos_ft)하지만 내부적으로 Chronos 2를 사용.
    """
    all_models = {
        "chronos_zero": lambda: ChronosImputation(fine_tune=False, context_len=context_len),
        "chronos_ft":   lambda: ChronosImputation(fine_tune=True,  context_len=context_len),
        "timesfm_zero": lambda: TimesFMImputation(context_len=context_len),
        "timesfm3_zero": lambda: TimesFM3Imputation(context_len=context_len),
        "timesfm3_mv":   lambda: TimesFM3MVImputation(context_len=context_len),
        "timesfm3_cov":  lambda: TimesFM3CovImputation(context_len=context_len),
        "cafi_chronos": lambda: CAFIImputation(fine_tune=True, context_len=context_len, name="CAFI-Chronos"),
        "cafi_timesfm3": lambda: CAFITimesFM3Imputation(context_len=context_len, name="CAFI-TimesFM3"),
        # 기존 CLI 호환용 alias.
        "cafi_ft":      lambda: CAFIImputation(fine_tune=True, context_len=context_len, name="CAFI-Chronos"),
    }

    if include is None:
        include = list(all_models.keys())

    result = []
    for key in include:
        if key not in all_models:
            warnings.warn(f"알 수 없는 모델 키: {key}")
            continue
        try:
            model = all_models[key]()
            result.append(model)
        except Exception as e:
            warnings.warn(f"{key} 초기화 실패: {e}")

    return result

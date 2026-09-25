"""
autogluon_model.py
AutoGluon Time Series 기반 보간 모델

전략:
  - 5개 변수를 각각 별도 item_id로 갖는 TimeSeriesDataFrame을 구성해 학습.
  - 보간 시 결측 블록(gap)마다 직전 context_len 포인트를 컨텍스트로 예측.
  - gap 길이가 prediction_length보다 길면 rolling prediction으로 확장.
  - autogluon import 실패 / predictor 미생성 시 선형 보간으로 fallback.

모델 전략 (hyperparameters 인자로 제어):
  - None (기본값)          : AutoGluon이 presets 기준으로 전체 모델 자동 선택·앙상블
  - 딕셔너리 명시          : 해당 모델만 학습 (e.g. {"Chronos": {}, "DeepAR": {}})

사용 가능한 주요 모델 키:
  Foundation : "Chronos", "Chronos2", "Toto"
  Neural     : "DeepAR", "DLinear", "PatchTST", "TiDE",
               "TemporalFusionTransformer", "WaveNet", "SimpleFeedForward"
  Statistical: "AutoARIMA", "AutoETS", "NPTS", "Theta", "DynamicOptimizedTheta"

presets 별 동작:
  "fast_training" (기본): 빠른 모델 위주 자동 선택, time_limit 60초
  "medium_quality"       : 중간 품질, 더 많은 모델 시도
  "best_quality"         : 전체 모델 + 앙상블, time_limit 600초 이상 권장
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .foundation_model import BaseImputationModel, DEFAULT_CONTEXT_LEN

warnings.filterwarnings("ignore")

# autogluon은 무겁고 선택적 의존성이므로 import 단계에서 try/except로 감싼다.
try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    _AUTOGLUON_AVAILABLE = True
    _AUTOGLUON_IMPORT_ERROR: Exception | None = None
except Exception as _e:  # ImportError 외에도 의존성 충돌 등을 모두 흡수
    TimeSeriesDataFrame = None  # type: ignore
    TimeSeriesPredictor = None  # type: ignore
    _AUTOGLUON_AVAILABLE = False
    _AUTOGLUON_IMPORT_ERROR = _e


class AutoGluonImputation(BaseImputationModel):
    """
    AutoGluon TimeSeriesPredictor 기반 보간.

    각 변수(Tin, Tout, RH, CO2, Rad)를 별도 item_id로 다루는
    long-format TimeSeriesDataFrame을 만들어 단일 predictor로 학습/예측한다.
    """

    def __init__(
        self,
        context_len: int = DEFAULT_CONTEXT_LEN,
        prediction_length: int = 48,
        time_limit: int = 600,
        presets: str = "best_quality",
        hyperparameters: dict | None = None,
        freq: str = "h",
        eval_metric: str = "MAE",
        name: str = "AutoGluon",
        save_path: str | Path | None = None,
        sub_model_name: str | None = None,
    ):
        self.context_len = context_len
        self.prediction_length = prediction_length
        self.time_limit = time_limit
        self.presets = presets
        # None → AutoGluon이 presets 기준으로 전체 모델을 자동 선택·앙상블한다.
        # 딕셔너리를 전달하면 해당 모델만 학습한다.
        self.hyperparameters = hyperparameters
        self.freq = freq
        self.eval_metric = eval_metric
        self.name = name
        # 학습 후 predictor와 리더보드를 저장할 경로. None이면 저장하지 않는다.
        self.save_path: Path | None = Path(save_path) if save_path else None
        # None이면 WeightedEnsemble(기본), 지정하면 해당 sub-model만 사용한다.
        self.sub_model_name: str | None = sub_model_name

        self.predictor: Any = None
        self._train_columns: list[str] | None = None

    # ── 내부 헬퍼 ──────────────────────────────

    def _make_tsdf(self, frame: pd.DataFrame) -> Any:
        """
        wide DataFrame(컬럼=변수) → long TimeSeriesDataFrame.
        각 변수는 별도 item_id가 되며, 결측 없는 연속 인덱스를 부여한다.
        """
        long_rows = []
        n = len(frame)
        # 학습/예측에서 동일한 freq의 균일한 timestamp를 사용한다.
        timestamps = pd.date_range("2020-01-01", periods=n, freq=self.freq)
        for col in frame.columns:
            vals = pd.Series(frame[col].values, dtype=np.float64)
            vals = vals.interpolate(method="linear", limit_direction="both").ffill().bfill().fillna(0.0)
            long_rows.append(
                pd.DataFrame(
                    {
                        "item_id": col,
                        "timestamp": timestamps,
                        "target": vals.values.astype(np.float64),
                    }
                )
            )
        long_df = pd.concat(long_rows, ignore_index=True)
        return TimeSeriesDataFrame.from_data_frame(
            long_df,
            id_column="item_id",
            timestamp_column="timestamp",
        )

    def _make_global_tsdf(self, frames: list[pd.DataFrame], names: list[str]) -> Any:
        """
        여러 온실의 wide DataFrame들을 풀링한 long TimeSeriesDataFrame.

        각 (온실, 변수)가 별도 item_id(f"{greenhouse}::{var}")가 된다.
        결손 변수(해당 온실에 없는 컬럼)는 자동으로 제외되므로, 변수 일부가
        부족한 온실도 보유 변수만으로 글로벌 학습에 기여한다(개선사항.md #5).
        타임스탬프는 온실별로 동일 freq의 균일 인덱스를 새로 부여한다.
        """
        long_rows = []
        for frame, gh in zip(frames, names):
            n = len(frame)
            if n == 0:
                continue
            timestamps = pd.date_range("2020-01-01", periods=n, freq=self.freq)
            for col in frame.columns:
                vals = pd.Series(frame[col].values, dtype=np.float64)
                vals = (
                    vals.interpolate(method="linear", limit_direction="both")
                    .ffill().bfill().fillna(0.0)
                )
                long_rows.append(
                    pd.DataFrame(
                        {
                            "item_id": f"{gh}::{col}",
                            "timestamp": timestamps,
                            "target": vals.values.astype(np.float64),
                        }
                    )
                )
        if not long_rows:
            raise ValueError("글로벌 학습용 데이터가 비었습니다.")
        long_df = pd.concat(long_rows, ignore_index=True)
        return TimeSeriesDataFrame.from_data_frame(
            long_df,
            id_column="item_id",
            timestamp_column="timestamp",
        )

    def _make_context_tsdf(self, item_id: str, context: np.ndarray) -> Any:
        """단일 변수의 컨텍스트 배열 → 예측용 TimeSeriesDataFrame."""
        ctx = pd.Series(context, dtype=np.float64)
        ctx = ctx.interpolate(method="linear", limit_direction="both").ffill().bfill().fillna(0.0)
        timestamps = pd.date_range("2020-01-01", periods=len(ctx), freq=self.freq)
        long_df = pd.DataFrame(
            {
                "item_id": item_id,
                "timestamp": timestamps,
                "target": ctx.values.astype(np.float64),
            }
        )
        return TimeSeriesDataFrame.from_data_frame(
            long_df,
            id_column="item_id",
            timestamp_column="timestamp",
        )

    def _persist_predictor(self) -> None:
        """모델을 메모리에 상주시켜 predict() 호출마다 디스크에서 재로드하는
        고정 오버헤드(특히 DeepAR/PatchTST의 torch 체크포인트 로드)를 제거한다.
        AutoGluon은 persist하지 않으면 매 predict마다 모델을 다시 로드한다."""
        if self.predictor is None:
            return
        try:
            persisted = self.predictor.persist(models="all")
            print(f"  [{self.name}] 모델 메모리 상주(persist): {persisted}")
        except Exception as e:
            warnings.warn(f"[{self.name}] persist 실패(디스크 로드로 동작, 느림): {e}")

    def _predict_once(self, item_id: str, context: np.ndarray, n_pred: int) -> np.ndarray:
        """컨텍스트로부터 n_pred(≤ prediction_length) 스텝을 예측."""
        ctx_tsdf = self._make_context_tsdf(item_id, context)
        predict_kwargs: dict = {}
        if self.sub_model_name is not None:
            predict_kwargs["model"] = self.sub_model_name
        pred = self.predictor.predict(ctx_tsdf, **predict_kwargs)
        # 반환 컬럼: 'mean' (+ quantile 컬럼들). point는 'mean' 우선, 없으면 첫 컬럼.
        if "mean" in pred.columns:
            point = pred["mean"].values
        else:
            point = pred.iloc[:, 0].values
        point = np.asarray(point, dtype=np.float32).flatten()[:n_pred]
        if len(point) < n_pred:
            point = np.pad(point, (0, n_pred - len(point)), mode="edge")
        return point

    def _rolling_predict(self, item_id: str, context: np.ndarray, n_pred: int) -> np.ndarray:
        """
        gap이 prediction_length보다 길면 rolling으로 채운다.
        예측값을 컨텍스트에 이어붙이며 prediction_length씩 반복 예측.
        """
        results: list[np.ndarray] = []
        cur_ctx = np.asarray(context, dtype=np.float32).copy()
        remaining = n_pred
        while remaining > 0:
            step = min(remaining, self.prediction_length)
            trimmed = cur_ctx[-self.context_len:] if len(cur_ctx) > self.context_len else cur_ctx
            chunk = self._predict_once(item_id, trimmed, step)
            results.append(chunk)
            cur_ctx = np.concatenate([cur_ctx, chunk])
            remaining -= step
        return np.concatenate(results)[:n_pred]

    # ── 공개 API ───────────────────────────────

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        if not _AUTOGLUON_AVAILABLE:
            warnings.warn(
                f"[{self.name}] autogluon 미설치/로드 실패 → 선형 보간으로 대체됩니다.\n"
                f"  원인: {_AUTOGLUON_IMPORT_ERROR}\n"
                "  pip install autogluon.timeseries"
            )
            self.predictor = None
            return

        try:
            self._train_columns = list(train_data.columns)
            train_tsdf = self._make_tsdf(train_data)

            # save_path가 있으면 predictor 초기화 시 경로를 지정한다.
            # AutoGluon은 path를 __init__에서 받고 save()는 인자 없이 호출한다.
            pred_init_path = (
                str(self.save_path / "predictor") if self.save_path else None
            )
            self.predictor = TimeSeriesPredictor(
                prediction_length=self.prediction_length,
                freq=self.freq,
                eval_metric=self.eval_metric,
                target="target",
                verbosity=1,
                **({"path": pred_init_path} if pred_init_path else {}),
            )
            fit_kwargs: dict = dict(
                presets=self.presets,
                time_limit=self.time_limit,
            )
            if self.hyperparameters is not None:
                fit_kwargs["hyperparameters"] = self.hyperparameters

            self.predictor.fit(train_tsdf, **fit_kwargs)

            trained = self.predictor.model_names()
            model_info = (
                list(self.hyperparameters.keys())
                if self.hyperparameters is not None
                else trained
            )
            print(
                f"  [{self.name}] 학습 완료 "
                f"(presets={self.presets}, time_limit={self.time_limit}s, "
                f"학습된 모델 {len(trained)}개: {model_info})"
            )

        except Exception as e:
            import traceback
            warnings.warn(f"  [{self.name}] fit 실패: {e} → 선형 보간으로 대체")
            traceback.print_exc()
            self.predictor = None
            return

        # 저장은 fit 성공 후 별도 try/except — 저장 실패가 predictor를 날리지 않는다.
        if self.save_path is not None:
            try:
                self.save_predictor(self.save_path)
            except Exception as e:
                warnings.warn(f"  [{self.name}] 저장 실패 (predictor는 유지): {e}")
        self._persist_predictor()

    def fit_global(self, train_frames: list[pd.DataFrame], names: list[str] | None = None, **kwargs) -> None:
        """
        여러 온실 데이터를 풀링해 단일 글로벌 predictor를 학습한다(개선사항.md #5).

        item_id를 f"{greenhouse}::{var}"로 확장해 한 predictor가 모든 온실·변수를
        학습하고, 예측은 Test 온실 컨텍스트(미관측 item)로 수행한다 — 글로벌 모델은
        미관측 series에도 일반화된다. 단일 fit()과 달리 다수 온실을 입력으로 받는 점만
        다르며, 저장/예측 경로는 동일하다.

        train_frames : 온실별 wide DataFrame 목록 (컬럼=변수, 결손 변수는 자동 제외)
        names        : 온실 ID 목록 (item_id prefix). None이면 인덱스로 자동 생성.
        """
        if not _AUTOGLUON_AVAILABLE:
            warnings.warn(
                f"[{self.name}] autogluon 미설치/로드 실패 → 선형 보간으로 대체됩니다.\n"
                f"  원인: {_AUTOGLUON_IMPORT_ERROR}"
            )
            self.predictor = None
            return

        if names is None:
            names = [f"gh{i:03d}" for i in range(len(train_frames))]

        try:
            # 단변량 보간 컨텍스트(impute)는 변수명 단위 item_id를 쓰므로,
            # _train_columns는 등장한 변수들의 합집합으로 둔다.
            all_cols: list[str] = []
            for f in train_frames:
                for c in f.columns:
                    if c not in all_cols:
                        all_cols.append(c)
            self._train_columns = all_cols

            train_tsdf = self._make_global_tsdf(train_frames, names)
            n_items = train_tsdf.item_ids.size if hasattr(train_tsdf, "item_ids") else None

            pred_init_path = (
                str(self.save_path / "predictor") if self.save_path else None
            )
            self.predictor = TimeSeriesPredictor(
                prediction_length=self.prediction_length,
                freq=self.freq,
                eval_metric=self.eval_metric,
                target="target",
                verbosity=1,
                **({"path": pred_init_path} if pred_init_path else {}),
            )
            fit_kwargs: dict = dict(presets=self.presets, time_limit=self.time_limit)
            if self.hyperparameters is not None:
                fit_kwargs["hyperparameters"] = self.hyperparameters

            validation_frames=kwargs.get("validation_frames")
            if validation_frames is not None:
                fit_kwargs.update(tuning_data=self._make_global_tsdf(validation_frames,names),num_val_windows=0,random_seed=42)
            self.predictor.fit(train_tsdf, **fit_kwargs)
            trained = self.predictor.model_names()
            print(
                f"  [{self.name}] 글로벌 학습 완료 "
                f"(온실 {len(train_frames)}개, item {n_items}개, "
                f"presets={self.presets}, time_limit={self.time_limit}s, 모델 {len(trained)}개)"
            )
        except Exception as e:
            import traceback
            warnings.warn(f"  [{self.name}] fit_global 실패: {e} → 선형 보간으로 대체")
            traceback.print_exc()
            self.predictor = None
            return

        if self.save_path is not None:
            try:
                self.save_predictor(self.save_path)
            except Exception as e:
                warnings.warn(f"  [{self.name}] 저장 실패 (predictor는 유지): {e}")
        self._persist_predictor()

    def impute(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.predictor is None:
            warnings.warn(f"[{self.name}] predictor 없음 → 선형 보간으로 대체")
            from .linear_interpolation import LinearInterpolation
            return LinearInterpolation().impute(masked_data, mask_matrix)

        imputed = masked_data.copy()

        for col in masked_data.columns:
            series = masked_data[col].copy()
            mask = mask_matrix[col].values  # 0=결측, 1=유효
            combined_missing = (mask == 0) | series.isna().values
            gaps = self._find_gaps(combined_missing)

            vals = series.values.copy().astype(np.float32)

            for gs, ge in gaps:
                n_pred = ge - gs + 1
                ctx_start = max(0, gs - self.context_len)
                context = vals[ctx_start:gs]
                context = (
                    pd.Series(context).ffill().bfill().fillna(0.0).values.astype(np.float32)
                )

                # 컨텍스트가 비면 gap 이후 값을 시간 역전해 사용한다.
                if len(context) == 0:
                    post_ctx = vals[ge + 1:ge + 1 + self.context_len]
                    post_ctx = (
                        pd.Series(post_ctx).ffill().bfill().fillna(0.0).values.astype(np.float32)
                    )
                    if len(post_ctx) == 0:
                        continue
                    context = post_ctx[::-1].copy()

                try:
                    if n_pred <= self.prediction_length:
                        pred = self._predict_once(col, context, n_pred)
                    else:
                        pred = self._rolling_predict(col, context, n_pred)
                    pred = np.asarray(pred, dtype=np.float32).flatten()[:n_pred]
                    if len(pred) < n_pred:
                        pred = np.pad(pred, (0, n_pred - len(pred)), mode="edge")
                    vals[gs:ge + 1] = pred
                except Exception as e:
                    if not getattr(self, "_error_reported", False):
                        print(
                            f"  [{self.name}] 예측 실패 (첫 발생): {e} "
                            "→ 해당 gap은 선형보간으로 대체"
                        )
                        self._error_reported = True

            imputed[col] = vals

        # 최종 정리: 남은 NaN을 선형 보간 + ffill/bfill로 채운다.
        imputed = imputed.interpolate(method="linear", limit_direction="both").ffill().bfill()
        return imputed

    # ── base 캐시 빠른 경로 ─────────────────────

    def compute_base(self, data: pd.DataFrame, valid_mask: pd.DataFrame) -> pd.DataFrame:
        """원본결측만 채운 전체 보간(온실당 1회 계산해 캐시)."""
        return self.impute(data, valid_mask)

    def _predict_batch(self, contexts: list[np.ndarray], step: int) -> list[np.ndarray]:
        """여러 컨텍스트를 하나의 multi-item TSDF로 묶어 predict()를 1회만 호출한다.

        predict() 호출당 고정 오버헤드가 크므로(시나리오 C·rolling에서 호출 수가
        곧 시간) item 단위 배치로 amortize한다. 각 컨텍스트는 별도 item_id가 된다.
        """
        frames = []
        for k, ctx in enumerate(contexts):
            c = pd.Series(ctx, dtype=np.float64)
            c = c.interpolate(method="linear", limit_direction="both").ffill().bfill().fillna(0.0)
            ts = pd.date_range("2020-01-01", periods=len(c), freq=self.freq)
            frames.append(pd.DataFrame({"item_id": f"job{k}", "timestamp": ts,
                                        "target": c.values.astype(np.float64)}))
        long_df = pd.concat(frames, ignore_index=True)
        tsdf = TimeSeriesDataFrame.from_data_frame(
            long_df, id_column="item_id", timestamp_column="timestamp"
        )
        predict_kwargs: dict = {}
        if self.sub_model_name is not None:
            predict_kwargs["model"] = self.sub_model_name
        pred = self.predictor.predict(tsdf, **predict_kwargs)

        out: list[np.ndarray] = []
        for k in range(len(contexts)):
            sub = pred.loc[f"job{k}"]
            point = sub["mean"].values if "mean" in sub.columns else sub.iloc[:, 0].values
            point = np.asarray(point, dtype=np.float32).flatten()[:step]
            if len(point) < step:
                point = np.pad(point, (0, step - len(point)), mode="edge")
            out.append(point)
        return out

    def impute_artificial(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
        artificial_bool: pd.DataFrame,
        base: pd.DataFrame,
    ) -> pd.DataFrame:
        """base에서 인공 gap만 재예측(원본결측은 base 재사용). 컬럼 간 배치 예측."""
        if self.predictor is None:
            return base[list(masked_data.columns)].copy()
        return self._impute_artificial_core(
            masked_data, mask_matrix, artificial_bool, base,
            self._predict_batch, self.prediction_length, self.context_len,
        )

    # ── 저장 / 로드 ────────────────────────────

    def create_submodel_wrappers(self) -> list["AutoGluonImputation"]:
        """
        학습된 predictor의 각 sub-model(앙상블 제외)에 대한 wrapper 목록을 반환한다.
        모든 wrapper는 학습된 predictor를 공유하므로 재학습이 필요 없다.
        """
        if self.predictor is None:
            return []
        wrappers = []
        for mname in self.predictor.model_names():
            if "Ensemble" in mname or "Weighted" in mname:
                continue
            wrapper = AutoGluonImputation(
                context_len=self.context_len,
                prediction_length=self.prediction_length,
                freq=self.freq,
                eval_metric=self.eval_metric,
                name=f"AG-{mname}",
                sub_model_name=mname,
            )
            wrapper.predictor = self.predictor
            wrapper._train_columns = self._train_columns
            wrappers.append(wrapper)
        return wrappers

    def get_leaderboard(self) -> pd.DataFrame:
        """
        학습된 모델의 리더보드와 앙상블 가중치를 반환한다.

        컬럼: model | score_val | pred_time_val | fit_time_marginal |
              fit_order | ensemble_weight
        ensemble_weight: WeightedEnsemble이 각 베이스 모델에 부여한 가중치.
                         앙상블에 포함되지 않은 모델은 0.0.
        """
        if self.predictor is None:
            return pd.DataFrame()

        lb = self.predictor.leaderboard(silent=True).copy()

        # info()의 model_info에서 WeightedEnsemble의 model_weights를 직접 추출한다.
        ensemble_weights: dict[str, float] = {}
        try:
            info = self.predictor.info()
            for mname, minfo in info.get("model_info", {}).items():
                if "Ensemble" in mname and isinstance(minfo.get("model_weights"), dict):
                    for base, w in minfo["model_weights"].items():
                        ensemble_weights[base] = float(w)
        except Exception:
            pass

        lb["ensemble_weight"] = lb["model"].map(ensemble_weights).fillna(0.0)
        return lb.sort_values("score_val", ascending=False).reset_index(drop=True)

    def save_predictor(self, save_dir: str | Path) -> None:
        """
        Predictor 전체와 리더보드(앙상블 가중치 포함)를 디스크에 저장한다.

        save_dir/
          predictor/      ← AutoGluon predictor (재로드 가능)
          leaderboard.csv ← 모델별 검증 점수 + 앙상블 가중치
        """
        if self.predictor is None:
            warnings.warn(f"[{self.name}] predictor 없음 → 저장 스킵")
            return

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 리더보드는 save() 이전에 가져온다 — save() 호출 후 내부 상태가 바뀔 수 있음
        lb = self.get_leaderboard()

        # predictor 저장 (AutoGluon은 초기화 시 지정한 path에 저장)
        pred_dir = save_dir / "predictor"
        self.predictor.save()  # path는 __init__에서 이미 지정됨
        lb_path = save_dir / "leaderboard.csv"
        lb.to_csv(lb_path, index=False, encoding="utf-8-sig")

        print(f"  [{self.name}] 저장 완료: {save_dir}")
        print(f"    predictor → {pred_dir}")
        print(f"    leaderboard ({len(lb)}개 모델) → {lb_path}")
        if "ensemble_weight" in lb.columns:
            top = lb[lb["ensemble_weight"] > 0].sort_values(
                "ensemble_weight", ascending=False
            )[["model", "ensemble_weight"]].head(5)
            if not top.empty:
                print("    앙상블 가중치 TOP 5:")
                for _, row in top.iterrows():
                    print(f"      {row['model']}: {row['ensemble_weight']:.4f}")

    @classmethod
    def load(cls, save_dir: str | Path, **init_kwargs) -> "AutoGluonImputation":
        """
        save_predictor()로 저장한 predictor를 재로드한다.
        재학습 없이 바로 impute() 사용 가능.
        """
        if not _AUTOGLUON_AVAILABLE:
            raise RuntimeError("autogluon 미설치. pip install autogluon.timeseries")

        save_dir = Path(save_dir)
        pred_dir = save_dir / "predictor"
        if not pred_dir.exists():
            raise FileNotFoundError(f"predictor 디렉터리 없음: {pred_dir}")

        instance = cls(**init_kwargs)
        instance.predictor = TimeSeriesPredictor.load(str(pred_dir))
        instance._persist_predictor()
        print(f"  [{instance.name}] predictor 로드 완료: {pred_dir}")
        return instance


class AutoGluonEnsembleImputation(AutoGluonImputation):
    """
    AutoGluon 다중 모델 앙상블 보간 (best_quality).

    Chronos + DeepAR + ETS + NPTS를 함께 학습하고 AutoGluon이 가중 앙상블을 구성한다.
    그 외 동작(보간 로직)은 AutoGluonImputation과 동일.
    """

    def __init__(
        self,
        context_len: int = DEFAULT_CONTEXT_LEN,
        prediction_length: int = 48,
        time_limit: int = 600,
        presets: str = "best_quality",
        hyperparameters: dict | None = None,
        freq: str = "h",
        eval_metric: str = "MAE",
        name: str = "AutoGluon-Ensemble",
    ):
        super().__init__(
            context_len=context_len,
            prediction_length=prediction_length,
            time_limit=time_limit,
            presets=presets,
            hyperparameters=(
                hyperparameters
                if hyperparameters is not None
                else {"Chronos": {}, "DeepAR": {}, "ETS": {}, "NPTS": {}}
            ),
            freq=freq,
            eval_metric=eval_metric,
            name=name,
        )

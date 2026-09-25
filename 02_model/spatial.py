"""
spatial.py
공간축(cross-site) 정보원 — 이웃 온실의 동시각 관측을 covariate로 활용한다.

동기:
  34개 온실의 동시각 상관을 보면 외기 구동 변수는 사실상 같은 신호다.
    Tout 최고이웃 r=0.991 | Rad 0.974 | Tin 0.943 | RH 0.790 | CO2 0.768
  기존 정보원(시간축=자기 과거, 변수축=같은 온실 다른 센서)은 해당 온실의
  고장 규모가 커질수록 말라붙지만(시나리오 A→C), 공간축은 이웃이 멀쩡하므로
  고장 양상과 무관하게 일정하다.

누수 차단 3원칙:
  1. 이웃 후보는 **train 온실만** (test-test 정보 공유 차단)
  2. 이웃 선택(상관)과 회귀 적합은 타깃의 **관측 구간에서만** 수행
     — 인공 gap 구간의 타깃 값은 어떤 경로로도 참조하지 않는다
  3. gap 구간에서 이웃 값은 **이웃 온실이 실제 관측한 값**만 사용
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_K = 3
MIN_COVERAGE = 0.8      # 이웃 후보가 타깃 기간을 덮어야 하는 최소 비율
MIN_FIT_POINTS = 500    # 상관/회귀 적합에 필요한 최소 관측 시점 수


class NeighborBank:
    """train 온실들의 정규화 시계열을 보관하고 타깃에 정렬해 제공한다."""

    def __init__(self, train_paths: list[Path], target_vars: list[str]):
        from preprocessing import preprocess_file

        self.target_vars = list(target_vars)
        self.sites: dict[str, pd.DataFrame] = {}
        for fp in train_paths:
            try:
                res = preprocess_file(fp, include_covariates=False)
            except Exception as e:
                warnings.warn(f"[NeighborBank] {fp.name} 로드 실패: {e}")
                continue
            if res is None:
                continue
            d = res["data"]
            cols = [c for c in self.target_vars if c in d.columns]
            if not cols:
                continue
            self.sites[res["name"]] = d[cols]
        print(f"  [NeighborBank] train 온실 {len(self.sites)}개 적재")

    # ── 이웃 정렬 ─────────────────────────────
    def aligned(self, var: str, index: pd.Index) -> pd.DataFrame:
        """변수 var에 대해 이웃 온실들을 타깃 시간축에 정렬한 (T, n_site) 프레임."""
        cols = {}
        n = len(index)
        for name, df in self.sites.items():
            if var not in df.columns:
                continue
            s = df[var].reindex(index)
            if s.notna().sum() / max(n, 1) < MIN_COVERAGE:
                continue
            cols[name] = s
        if not cols:
            return pd.DataFrame(index=index)
        return pd.DataFrame(cols, index=index)

    def select(
        self,
        var: str,
        index: pd.Index,
        target: np.ndarray,
        observed: np.ndarray,
        k: int = DEFAULT_K,
    ) -> tuple[np.ndarray, list[str]]:
        """
        타깃의 **관측 구간**에서만 상관을 계산해 상위 k개 이웃을 고른다.

        Returns
        -------
        (k, T) 배열 — 타깃 시간축에 정렬된 이웃 값(결측은 보간). 이웃이 없으면 (0, T).
        선택된 이웃 이름 목록.
        """
        cand = self.aligned(var, index)
        if cand.empty:
            return np.zeros((0, len(index)), dtype=np.float32), []

        fit = observed & np.isfinite(target)
        if fit.sum() < MIN_FIT_POINTS:
            return np.zeros((0, len(index)), dtype=np.float32), []

        y = pd.Series(target[fit])
        scores: list[tuple[float, str]] = []
        for name in cand.columns:
            x = cand[name].values[fit]
            m = np.isfinite(x)
            if m.sum() < MIN_FIT_POINTS:
                continue
            r = pd.Series(x[m]).corr(y[m])
            if np.isfinite(r):
                scores.append((abs(float(r)), name))
        if not scores:
            return np.zeros((0, len(index)), dtype=np.float32), []

        scores.sort(reverse=True)
        picked = [n for _, n in scores[:k]]
        arr = cand[picked].interpolate(limit_direction="both").ffill().bfill()
        return arr.values.T.astype(np.float32), picked


# ──────────────────────────────────────────────
# 모델 1: 순수 공간 회귀 (통제용 baseline)
# ──────────────────────────────────────────────

class SpatialRidgeImputation:
    """
    파운데이션 모델 없이 이웃 온실 값만으로 gap을 채우는 능형회귀 baseline.

    존재 이유: "Tout 상관이 0.99인데 굳이 파운데이션 모델이 필요한가?"라는
    당연한 반문에 답하기 위한 통제군. 이 baseline이 이기는 변수/시나리오가
    있다면 그것도 논문의 정직한 발견이다.

    적합/예측:
      fit  : 타깃의 관측 시점에서 y ~ Ridge(이웃 k개)
      pred : gap 시점의 이웃 값으로 예측
      이웃이 없거나 적합 표본이 부족하면 선형보간으로 대체(fallback 기록).
    """

    def __init__(self, bank: NeighborBank, k: int = DEFAULT_K, alpha: float = 1.0):
        self.bank = bank
        self.k = k
        self.alpha = alpha
        self.name = "Spatial-Ridge"
        self.greenhouse: str | None = None
        self.stats = {"gaps": 0, "spatial": 0, "fallback": 0}

    def set_greenhouse(self, name: str) -> None:
        self.greenhouse = name

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        print(f"  [{self.name}] 학습형 아님 — gap마다 이웃 회귀를 새로 적합")

    @staticmethod
    def _find_gaps(missing: np.ndarray) -> list[tuple[int, int]]:
        gaps, start = [], None
        for i, m in enumerate(missing):
            if m and start is None:
                start = i
            elif not m and start is not None:
                gaps.append((start, i - 1)); start = None
        if start is not None:
            gaps.append((start, len(missing) - 1))
        return gaps

    def impute(self, masked_data: pd.DataFrame, mask_matrix: pd.DataFrame) -> pd.DataFrame:
        from sklearn.linear_model import Ridge

        out = masked_data.copy()
        index = masked_data.index
        for col in masked_data.columns:
            series = masked_data[col].values.astype(np.float32)
            observed = (mask_matrix[col].values != 0) & np.isfinite(series)
            nb, _ = self.bank.select(col, index, series, observed, k=self.k)

            vals = series.copy()
            missing = ~observed
            for gs, ge in self._find_gaps(missing):
                self.stats["gaps"] += 1
                if nb.shape[0] == 0:
                    self.stats["fallback"] += 1
                    continue
                # 적합은 '이 gap 밖의 관측 시점'만 사용한다.
                fit = observed.copy()
                fit[gs:ge + 1] = False
                if fit.sum() < MIN_FIT_POINTS:
                    self.stats["fallback"] += 1
                    continue
                X = nb[:, fit].T
                y = series[fit]
                ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
                if ok.sum() < MIN_FIT_POINTS:
                    self.stats["fallback"] += 1
                    continue
                try:
                    model = Ridge(alpha=self.alpha).fit(X[ok], y[ok])
                    Xg = nb[:, gs:ge + 1].T
                    if not np.isfinite(Xg).all():
                        Xg = np.nan_to_num(Xg, nan=float(np.nanmean(X[ok])))
                    vals[gs:ge + 1] = model.predict(Xg).astype(np.float32)
                    self.stats["spatial"] += 1
                except Exception:
                    self.stats["fallback"] += 1
            out[col] = vals

        return out.interpolate(method="linear", limit_direction="both").ffill().bfill()


# ──────────────────────────────────────────────
# 모델 2: TimesFM 3.0 — 변수축 + 공간축 covariate
# ──────────────────────────────────────────────

def make_tfm3_cov_spatial(bank: NeighborBank, context_len: int, k: int = DEFAULT_K):
    """
    TimesFM3CovImputation(변수축)에 공간축 covariate를 추가한 arm을 만든다.

    past_future_covariates 구성:
      [나머지 4개 변수] + [캘린더 4개] + [이웃 온실 k개의 같은 변수]
    앞의 8개는 기존 cov arm과 동일하므로, 성능 차이는 오직 공간축 k행에서 온다.
    """
    from models.foundation_model import TimesFM3CovImputation

    class TimesFM3CovSpatialImputation(TimesFM3CovImputation):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.name = "TimesFM3.0-COV-SPA"
            self.bank = bank
            self.k = k
            self.greenhouse: str | None = None
            self._nb_cache: dict[str, np.ndarray] = {}
            self._cur_index: pd.Index | None = None
            self.stats = {"with_spatial": 0, "no_neighbor": 0}

        def set_greenhouse(self, name: str) -> None:
            self.greenhouse = name
            self._nb_cache.clear()

        def _neighbors(self, col, index, target, observed) -> np.ndarray:
            """현재 마스크×변수별 이웃 선택을 캐시한다(숨긴 타깃은 선택에서 제외)."""
            if self._cur_index is None or not index.equals(self._cur_index):
                self._nb_cache.clear()
                self._cur_index = index
            if col not in self._nb_cache:
                nb, picked = self.bank.select(col, index, target, observed, k=self.k)
                self._nb_cache[col] = nb
                if nb.shape[0] == 0:
                    self.stats["no_neighbor"] += 1
            return self._nb_cache[col]

        def _impute_variant(self, masked_data, mask_matrix, estimate, artificial_bool=None):
            # 이웃 선택에 필요한 관측 정보를 미리 걸어둔다.
            self._nb_cache.clear()
            self._md, self._mm = masked_data, mask_matrix
            return super()._impute_variant(masked_data, mask_matrix, estimate, artificial_bool)

        def _predict_target(self, all_est, col_idx, gs, ge, index):
            n_pred = ge - gs + 1
            ctx_start = max(0, gs - self.context_len)
            target_ctx = all_est[ctx_start:gs, col_idx].astype(np.float32)
            if len(target_ctx) == 0:
                raise ValueError("empty context")

            col = list(self._md.columns)[col_idx]
            series = self._md[col].values.astype(np.float32)
            observed = (self._mm[col].values != 0) & np.isfinite(series)
            nb_full = self._neighbors(col, index, series, observed)

            other = [j for j in range(all_est.shape[1]) if j != col_idx]
            results, cur_ctx, pos, remaining = [], target_ctx.copy(), gs, n_pred
            while remaining > 0:
                chunk = min(remaining, self.horizon_len)
                t_ctx = cur_ctx[-self.context_len:] if len(cur_ctx) > self.context_len else cur_ctx
                c0 = pos - len(t_ctx)
                cov_past = all_est[c0:pos, :][:, other].T
                cov_fut = all_est[pos:pos + chunk, :][:, other].T
                pf = np.concatenate([cov_past, cov_fut], axis=1).astype(np.float32)
                if self.use_time_covariates:
                    pf = np.concatenate([pf, self._time_cov(index[c0:pos + chunk])], axis=0)
                if nb_full.shape[0] > 0:
                    # 공간축: 이웃의 과거 + gap 구간 실측값
                    pf = np.concatenate([pf, nb_full[:, c0:pos + chunk]], axis=0)
                    self.stats["with_spatial"] += 1
                if pf.shape[1] != len(t_ctx) + chunk:
                    raise ValueError(f"covariate 길이 불일치: {pf.shape[1]}")

                out = self.tfm.predict(context=t_ctx, horizon=chunk,
                                       past_future_covariates=pf, padding_mode="edge")
                pred = np.asarray(out.forecast, dtype=np.float32).flatten()[:chunk]
                results.append(pred)
                cur_ctx = np.concatenate([cur_ctx, pred])[-self.context_len:]
                pos += chunk
                remaining -= chunk
            return np.concatenate(results)[:n_pred]

    return TimesFM3CovSpatialImputation(context_len=context_len)

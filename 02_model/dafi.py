"""
BiTFI retrospective imputation (legacy module/result ID: DAFI).

Step 1 initializes missing targets with independent forward and reversed-backward
forecasts. Step 2 applies a configurable number of synchronous refinement passes
using the preceding-pass snapshot, calendar features and available same-variable
training-greenhouse observations. Set refinements=5 for the validation-selected
publication configuration; the default of one preserves the reference interface.
Directional predictions are aligned and fused with distance-dependent weights.
One-sided inference is used when only one context is available.

Hidden targets are masked before filling. Artificial-gap evaluation ignores
pre-mask base reconstructions, and neighbor state belongs to one mask only.
The forward-only arm removes the backward calls; other retrospective input
availability remains unchanged. Performance claims belong in evaluated results,
not in this algorithm description.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from models.foundation_model import (
    BaseImputationModel,
    DEFAULT_CONTEXT_LEN,
    _CHRONOS_PIPELINE_CACHE,
    _TIMESFM_BACKEND_CACHE,
)
from spatial import NeighborBank, DEFAULT_K

MIN_SIDE_CTX = 48        # 한쪽 컨텍스트가 이보다 짧으면 그 방향 pass를 생략한다


class DAFIImputation(BaseImputationModel):
    """백본 무관 공통 로직. 서브클래스가 _fc_uni / _fc_cov / _max_h 를 구현한다."""

    def __init__(
        self,
        bank: NeighborBank | None = None,
        context_len: int = DEFAULT_CONTEXT_LEN,
        k: int = DEFAULT_K,
        use_spatial: bool = True,
        use_time_covariates: bool = True,
        bidirectional: bool = True,
        name: str = "DAFI",
        refinements: int = 1,
    ):
        self.bank = bank
        self.context_len = context_len
        self.k = k
        self.use_spatial = use_spatial and bank is not None
        self.use_time_covariates = use_time_covariates
        self.bidirectional = bidirectional
        self.name = name
        if refinements < 0: raise ValueError("refinements must be nonnegative")
        self.refinements = int(refinements)
        self.greenhouse: str | None = None
        self._nb_cache: dict[str, np.ndarray] = {}
        self.uses_ext_covariates = False
        self.stats = {"gaps": 0, "bidir": 0, "fwd_only": 0, "bwd_only": 0, "too_long": 0}

    # ── 백엔드 훅 ─────────────────────────────
    def _fc_uni(self, ctx: np.ndarray, h: int) -> np.ndarray:
        raise NotImplementedError

    def _fc_cov(self, ctx: np.ndarray, names: list[str], pf: np.ndarray, h: int) -> np.ndarray:
        """pf: (n_cov, len(ctx)+h). 앞 len(ctx)열은 과거, 뒤 h열은 gap 구간."""
        raise NotImplementedError

    def _max_h(self) -> int:
        raise NotImplementedError

    def _ready(self) -> bool:
        raise NotImplementedError

    # ── 공통 유틸 ────────────────────────────
    def set_greenhouse(self, name: str) -> None:
        self.greenhouse = name
        self._nb_cache.clear()

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        print(f"  [{self.name}] zero-shot — 학습 생략")

    @staticmethod
    def _time_cov(index: pd.Index) -> np.ndarray:
        dt = pd.DatetimeIndex(index)
        hour = dt.hour.to_numpy(dtype=np.float32)
        doy = dt.dayofyear.to_numpy(dtype=np.float32)
        return np.stack([
            np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        ]).astype(np.float32)

    @staticmethod
    def _clean(x: np.ndarray) -> np.ndarray:
        return pd.Series(x).ffill().bfill().fillna(0.0).values.astype(np.float32)

    @staticmethod
    def _fuse(fwd: np.ndarray | None, bwd: np.ndarray | None) -> np.ndarray:
        if fwd is None:
            return bwd
        if bwd is None:
            return fwd
        n = len(fwd)
        w = np.linspace(1.0, 0.0, n, dtype=np.float32) if n > 1 else np.array([0.5], dtype=np.float32)
        return w * fwd + (1.0 - w) * bwd

    def _sides(self, col_vals: np.ndarray, gs: int, ge: int) -> tuple[np.ndarray, np.ndarray]:
        left = self._clean(col_vals[max(0, gs - self.context_len):gs])
        right = self._clean(col_vals[ge + 1:ge + 1 + self.context_len])
        return left, right

    # ── R0: 양방향 단변량 ─────────────────────
    def _gap_uni(self, col_vals: np.ndarray, gs: int, ge: int) -> np.ndarray | None:
        h = ge - gs + 1
        if h > self._max_h():
            self.stats["too_long"] += 1
            return None
        left, right = self._sides(col_vals, gs, ge)
        fwd = self._fc_uni(left, h) if len(left) >= MIN_SIDE_CTX else None
        bwd = None
        if self.bidirectional and len(right) >= MIN_SIDE_CTX:
            bwd = self._fc_uni(right[::-1].copy(), h)[::-1].copy()
        if fwd is None and bwd is None:
            if len(left) > 0:
                fwd = self._fc_uni(left, h)
            elif len(right) > 0:
                bwd = self._fc_uni(right[::-1].copy(), h)[::-1].copy()
            else:
                return None
        return self._fuse(fwd, bwd)

    # ── R1: 양방향 covariate ──────────────────
    def _cov_block(self, est: np.ndarray, others: list[int], nb: np.ndarray | None,
                   index: pd.Index, lo: int, hi: int) -> tuple[list[str], np.ndarray]:
        """[lo, hi) 구간의 covariate 행렬 (n_cov, hi-lo)과 이름."""
        names = [f"v{j}" for j in others]
        rows = [est[lo:hi, others].T]
        if self.use_time_covariates:
            names += ["t_hs", "t_hc", "t_ds", "t_dc"]
            rows.append(self._time_cov(index[lo:hi]))
        if nb is not None and nb.shape[0] > 0:
            names += [f"nb{i}" for i in range(nb.shape[0])]
            rows.append(nb[:, lo:hi])
        return names, np.concatenate(rows, axis=0).astype(np.float32)

    def _gap_cov(self, est: np.ndarray, col: int, gs: int, ge: int,
                 index: pd.Index, nb: np.ndarray | None) -> np.ndarray | None:
        h = ge - gs + 1
        if h > self._max_h():
            return None
        others = [j for j in range(est.shape[1]) if j != col]
        left, right = self._sides(est[:, col], gs, ge)

        fwd = bwd = None
        if len(left) >= MIN_SIDE_CTX:
            lo = gs - len(left)
            names, pf = self._cov_block(est, others, nb, index, lo, ge + 1)
            fwd = self._fc_cov(left, names, pf, h)
        if self.bidirectional and len(right) >= MIN_SIDE_CTX:
            hi = ge + 1 + len(right)
            names, blk = self._cov_block(est, others, nb, index, gs, hi)
            pf_b = blk[:, ::-1].copy()                 # 시간 반전: 우측 컨텍스트가 '과거', gap이 '미래'
            bwd = self._fc_cov(right[::-1].copy(), names, pf_b, h)[::-1].copy()
        if fwd is None and bwd is None:
            return None
        self.stats["bidir" if (fwd is not None and bwd is not None) else
                   ("fwd_only" if fwd is not None else "bwd_only")] += 1
        return self._fuse(fwd, bwd)

    # ── 핵심 실행 ────────────────────────────
    def _neighbors(self, col: str, index: pd.Index, series: np.ndarray, observed: np.ndarray):
        if not self.use_spatial:
            return None
        if col not in self._nb_cache:
            nb, _ = self.bank.select(col, index, series, observed, k=self.k)
            self._nb_cache[col] = nb
        return self._nb_cache[col]

    def _run(self, masked: pd.DataFrame, mask: pd.DataFrame,
             base: pd.DataFrame | None, artificial: pd.DataFrame | None) -> pd.DataFrame:
        self._nb_cache.clear()
        masked = masked.mask(mask.eq(0))
        cols = list(masked.columns)
        index = masked.index
        obs = masked[cols].values.astype(np.float32)
        observed = (mask[cols].values != 0) & np.isfinite(obs)

        # base는 현재 마스크 입력에서 계산한 값만 내부적으로 허용한다.
        if base is not None:
            est = base[cols].values.astype(np.float32).copy()
        else:
            est = masked[cols].interpolate(limit_direction="both").ffill().bfill().values.astype(np.float32)
        est[observed] = obs[observed]

        # 처리할 gap 목록
        jobs: list[tuple[int, int, int]] = []
        for ci, c in enumerate(cols):
            combined = ~observed[:, ci]
            for gs, ge in self._find_gaps(combined):
                if artificial is not None:
                    if c not in artificial.columns or not artificial[c].values[gs:ge + 1].any():
                        continue
                jobs.append((ci, gs, ge))
        self.stats["gaps"] += len(jobs)
        if not jobs:
            return pd.DataFrame(est, index=index, columns=cols)

        # 초기화된 인공 gap 위치는 R0 예측으로 대체한다.
        for ci, gs, ge in jobs:
            est[gs:ge + 1, ci] = np.nan

        # R0: 양방향 단변량 (자기 컬럼의 좌·우 컨텍스트만 사용)
        for ci, gs, ge in jobs:
            colv = est[:, ci].copy()
            p = self._gap_uni(colv, gs, ge)
            if p is None:
                s = pd.Series(colv); s.iloc[gs:ge + 1] = np.nan
                est[gs:ge + 1, ci] = s.interpolate(limit_direction="both").ffill().bfill().values[gs:ge + 1]
            else:
                est[gs:ge + 1, ci] = p[:ge - gs + 1]

        # Every refinement reads one fixed preceding-pass snapshot.
        for _ in range(self.refinements):
            out = est.copy()
            for ci, gs, ge in jobs:
                nb = self._neighbors(cols[ci], index, obs[:, ci], observed[:, ci])
                p = self._gap_cov(est, ci, gs, ge, index, nb)
                if p is not None:
                    out[gs:ge + 1, ci] = p[:ge - gs + 1]
            est = out
        res = pd.DataFrame(est, index=index, columns=cols)
        return res.interpolate(method="linear", limit_direction="both").ffill().bfill()

    # ── 공개 인터페이스 ───────────────────────
    def impute(self, masked_data: pd.DataFrame, mask_matrix: pd.DataFrame) -> pd.DataFrame:
        if not self._ready():
            from models.linear_interpolation import LinearInterpolation
            return LinearInterpolation().impute(masked_data, mask_matrix)
        return self._run(masked_data, mask_matrix, base=None, artificial=None)

    def compute_base(self, data: pd.DataFrame, valid_mask: pd.DataFrame) -> pd.DataFrame:
        """원본결측만 채운 전체 보간 (R0 양방향 단변량; 평가 대상 아님)."""
        if not self._ready():
            return data.interpolate(limit_direction="both").ffill().bfill()
        cols = list(data.columns)
        obs = data[cols].values.astype(np.float32)
        observed = (valid_mask[cols].values != 0) & np.isfinite(obs)
        est = data[cols].interpolate(limit_direction="both").ffill().bfill().values.astype(np.float32)
        est[observed] = obs[observed]
        for ci in range(len(cols)):
            for gs, ge in self._find_gaps(~observed[:, ci]):
                p = self._gap_uni(est[:, ci].copy(), gs, ge)
                if p is not None:
                    est[gs:ge + 1, ci] = p[:ge - gs + 1]
        return pd.DataFrame(est, index=data.index, columns=cols)

    def impute_artificial(self, masked_data, mask_matrix, artificial_bool, base):
        if not self._ready():
            raise RuntimeError(f"{self.name}: backend unavailable; refusing truth-bearing fallback")
        self._nb_cache.clear()
        # Never trust a pre-mask base. Reconstruct context solely from masked inputs.
        safe = masked_data.mask(mask_matrix.eq(0) | artificial_bool)
        safe_base = safe.interpolate(limit_direction="both").ffill().bfill()
        return self._run(safe, mask_matrix, base=safe_base, artificial=artificial_bool)


# ──────────────────────────────────────────────
# 백엔드 1: TimesFM 3.0
# ──────────────────────────────────────────────

class DAFITimesFM3(DAFIImputation):
    def __init__(self, model_id: str = "google/timesfm-3.0-pytorch", **kw):
        super().__init__(**kw)
        self.model_id = model_id
        self.tfm: Any = None
        try:
            import timesfm, torch
            torch.set_float32_matmul_precision("high")
            if not hasattr(timesfm, "TimesFM3Forecaster"):
                raise ImportError("timesfm>=3.0.1 필요")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            key = (model_id, "dafi", self.context_len)
            if key not in _TIMESFM_BACKEND_CACHE:
                _TIMESFM_BACKEND_CACHE[key] = timesfm.TimesFM3Forecaster.from_pretrained(model_id, device=device)
                print(f"  [{self.name}] TimesFM 3.0 로드 완료 (device={device})")
            self.tfm = _TIMESFM_BACKEND_CACHE[key]
        except Exception as e:
            warnings.warn(f"[{self.name}] TimesFM 3.0 로드 실패: {e}")

    def _ready(self) -> bool:
        return self.tfm is not None

    def _max_h(self) -> int:
        return 2048

    def _fc_uni(self, ctx, h):
        import torch
        with torch.inference_mode():
            o = self.tfm.predict(context=np.asarray(ctx, dtype=np.float32), horizon=h)
        return np.asarray(o.forecast, dtype=np.float32).flatten()[:h]

    def _fc_cov(self, ctx, names, pf, h):
        import torch
        with torch.inference_mode():
            o = self.tfm.predict(context=np.asarray(ctx, dtype=np.float32), horizon=h,
                                 past_future_covariates=pf, padding_mode="edge")
        return np.asarray(o.forecast, dtype=np.float32).flatten()[:h]


# ──────────────────────────────────────────────
# 백엔드 2: Chronos-2
# ──────────────────────────────────────────────

class DAFIChronos2(DAFIImputation):
    def __init__(self, model_id: str = "amazon/chronos-2", **kw):
        super().__init__(**kw)
        self.model_id = model_id
        self.pipeline: Any = None
        try:
            import torch
            try:
                from chronos import Chronos2Pipeline as _P
            except ImportError:
                from chronos import BaseChronosPipeline as _P
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float32
            key = f"{model_id}|{device}"
            if key not in _CHRONOS_PIPELINE_CACHE:
                _CHRONOS_PIPELINE_CACHE[key] = _P.from_pretrained(model_id, device_map=device, dtype=dtype)
                print(f"  [{self.name}] Chronos-2 로드 완료 (device={device})")
            self.pipeline = _CHRONOS_PIPELINE_CACHE[key]
        except Exception as e:
            warnings.warn(f"[{self.name}] Chronos-2 로드 실패: {e}")

    def _ready(self) -> bool:
        return self.pipeline is not None

    def _max_h(self) -> int:
        return 1024

    def _fc_uni(self, ctx, h):
        import torch
        with torch.inference_mode():
            _, mean = self.pipeline.predict_quantiles(
                inputs=[torch.tensor(np.asarray(ctx, dtype=np.float32))],
                prediction_length=h, quantile_levels=[0.1, 0.5, 0.9])
        return mean[0].squeeze(0).float().cpu().numpy().astype(np.float32)[:h]

    def _fc_cov(self, ctx, names, pf, h):
        import torch
        n_ctx = len(ctx)
        past = {n: torch.tensor(pf[i, :n_ctx].copy()) for i, n in enumerate(names)}
        fut = {n: torch.tensor(pf[i, n_ctx:n_ctx + h].copy()) for i, n in enumerate(names)}
        with torch.inference_mode():
            fc = self.pipeline.predict(
                inputs=[{"target": torch.tensor(np.asarray(ctx, dtype=np.float32)),
                         "past_covariates": past, "future_covariates": fut}],
                prediction_length=h)
        t = fc[0]
        if t.ndim == 3:
            return t[0, t.shape[1] // 2, :].float().cpu().numpy().astype(np.float32)[:h]
        return t.float().cpu().numpy().flatten().astype(np.float32)[:h]

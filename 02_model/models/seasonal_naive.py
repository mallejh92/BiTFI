"""
seasonal_naive.py
Seasonal Naive 보간 모델

전략:
  결측 위치 t에서, t - season_length * k (k=1,2,...) 중
  context window 내에 유효 관측값이 있는 가장 가까운 과거 계절 주기 값을 사용.

  - season_length = 24: 하루 주기 (24 h, 온실의 기본 일주기)
  - context window: 마스크 바로 앞 context_len_h 시간 내부만 참조
    (모든 시퀀스 모델과 동일 조건 적용)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .foundation_model import BaseImputationModel, DEFAULT_CONTEXT_LEN
except ImportError:
    from foundation_model import BaseImputationModel, DEFAULT_CONTEXT_LEN


class SeasonalNaiveImputation(BaseImputationModel):
    """
    Seasonal Naive 보간.

    각 결측 위치에 대해 season_length의 배수만큼 과거를 탐색하여
    context window 내 유효 관측값을 가져온다.
    유효 관측값이 없으면 선형 보간으로 fallback.
    """

    def __init__(
        self,
        season_length: int = 24,
        context_len: int = DEFAULT_CONTEXT_LEN,
        name: str = "SeasonalNaive",
    ):
        self.season_length = season_length
        self.context_len = context_len
        self.name = name

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        pass  # 학습 없음

    def impute(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        imputed = masked_data.copy()

        for col in masked_data.columns:
            series = masked_data[col].values.copy().astype(np.float64)
            mask = mask_matrix[col].values.astype(np.float32)

            combined_missing = (mask == 0) | np.isnan(series)
            gaps = self._find_gaps(combined_missing)

            for gs, ge in gaps:
                for t in range(gs, ge + 1):
                    # context window 시작점
                    ctx_start = max(0, gs - self.context_len)

                    # season_length 배수만큼 과거 탐색
                    filled = False
                    candidate = t - self.season_length
                    while candidate >= ctx_start:
                        if (
                            candidate >= 0
                            and not np.isnan(series[candidate])
                            and mask[candidate] == 1
                        ):
                            series[t] = series[candidate]
                            filled = True
                            break
                        candidate -= self.season_length

                    # 역방향(gap 이후)도 시도 — gap 앞 context가 없는 경우
                    if not filled:
                        candidate = t + self.season_length
                        ctx_end = min(len(series), ge + 1 + self.context_len)
                        while candidate < ctx_end:
                            if (
                                candidate < len(series)
                                and not np.isnan(series[candidate])
                                and mask[candidate] == 1
                            ):
                                series[t] = series[candidate]
                                break
                            candidate += self.season_length

            imputed[col] = series

        # 남은 NaN은 선형 보간으로 처리
        imputed = imputed.interpolate(method="linear", limit_direction="both").ffill().bfill()
        return imputed

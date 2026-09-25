"""
linear_interpolation.py
가장 단순한 baseline: pandas 선형 보간 + forward/backward fill
"""

import numpy as np
import pandas as pd


class LinearInterpolation:
    """
    Baseline 보간 모델.

    - 개별 결측: 선형 보간 (pandas interpolate)
    - 블록 결측(연속 긴 구간): forward fill → backward fill 후처리
    """

    name = "LI"

    def fit(self, train_data: pd.DataFrame) -> None:
        """학습 없음 — 인터페이스 통일을 위해 존재"""
        pass

    def impute(
        self,
        masked_data: pd.DataFrame,
        mask_matrix: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        결측 위치를 선형 보간으로 채움.

        Parameters
        ----------
        masked_data  : 결측이 NaN인 DataFrame
        mask_matrix  : 0(결측)/1(유효) DataFrame

        Returns
        -------
        imputed : 결측 위치가 채워진 DataFrame
        """
        imputed = masked_data.copy()

        for col in masked_data.columns:
            # 1차: 선형 보간 (limit 없이 전체)
            imputed[col] = imputed[col].interpolate(
                method="linear", limit_direction="both"
            )
            # 2차: 블록 결측 등 보간 후 남은 NaN → ffill → bfill
            imputed[col] = imputed[col].ffill().bfill()

        return imputed

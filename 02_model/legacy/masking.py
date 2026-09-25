"""
masking.py
논문 Moon et al. (2021) 의 두 가지 마스킹 전략 구현

(A) Random individual loss: 변수별 독립 랜덤 결측
(B) All-sensor block loss: 모든 변수 동시 연속 결측
"""

import numpy as np
import pandas as pd


def _get_block_points(freq: str, hours: int = 48) -> int:
    """freq 문자열로부터 N시간에 해당하는 포인트 수 계산"""
    freq_to_min = {
        "1min": 1,
        "5min": 5,
        "10min": 10,
        "1H": 60,
    }
    minutes_per_point = freq_to_min.get(freq, 60)
    return int(hours * 60 / minutes_per_point)


def create_random_individual_mask(
    df: pd.DataFrame,
    loss_rate: float = 0.30,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    (A) Random individual loss
    각 변수에 독립적으로 랜덤 결측 적용.

    Parameters
    ----------
    df : 정규화된 DataFrame (결측 없는 구간이 평가 대상)
    loss_rate : 결측 비율 (0 ~ 1)
    random_seed : 재현성 보장

    Returns
    -------
    masked_data  : 결측 위치가 NaN인 DataFrame
    mask_matrix  : 0(결측)/1(유효) DataFrame (U-Net 입력용)
    ground_truth : 원본 DataFrame (평가용)
    """
    rng = np.random.RandomState(random_seed)
    ground_truth = df.copy()
    masked_data = df.copy()
    mask_matrix = pd.DataFrame(
        np.ones(df.shape, dtype=np.float32),
        index=df.index,
        columns=df.columns,
    )

    for col in df.columns:
        valid_idx = df[col].dropna().index
        n_valid = len(valid_idx)
        n_mask = int(n_valid * loss_rate)
        if n_mask == 0:
            continue
        chosen = rng.choice(n_valid, size=n_mask, replace=False)
        mask_idx = valid_idx[chosen]
        masked_data.loc[mask_idx, col] = np.nan
        mask_matrix.loc[mask_idx, col] = 0.0

    return masked_data, mask_matrix, ground_truth


def create_block_loss_mask(
    df: pd.DataFrame,
    loss_rate: float = 0.30,
    freq: str = "1H",
    block_hours: int = 48,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    (B) All-sensor block loss
    모든 변수가 동시에 연속으로 결측되는 블록을 무작위 배치.

    block_length는 block_hours 기준으로 freq에 따라 자동 계산.

    Returns
    -------
    masked_data  : 결측 위치가 NaN인 DataFrame
    mask_matrix  : 0(결측)/1(유효) DataFrame
    ground_truth : 원본 DataFrame
    """
    rng = np.random.RandomState(random_seed)
    ground_truth = df.copy()
    masked_data = df.copy()
    mask_matrix = pd.DataFrame(
        np.ones(df.shape, dtype=np.float32),
        index=df.index,
        columns=df.columns,
    )

    n = len(df)
    block_len = max(1, _get_block_points(freq, hours=block_hours))
    target_masked = min(n, int(round(n * loss_rate)))

    # 목표 결측률을 채우기 위해 잔여 길이에 맞춰 블록 길이를 유연하게 줄이며 배치한다.
    occupied = np.zeros(n, dtype=bool)
    remaining = target_masked

    while remaining > 0:
        cur_len = min(block_len, remaining)
        placed = False

        while cur_len >= 1:
            # numpy convolution으로 각 위치의 cur_len 윈도우 내 occupied 합 계산 → O(n)
            conv = np.convolve(occupied.astype(np.uint8), np.ones(cur_len, dtype=np.uint8), mode="valid")
            valid_starts = np.where(conv == 0)[0]
            if len(valid_starts) > 0:
                start = int(rng.choice(valid_starts))
                end = start + cur_len
                occupied[start:end] = True
                remaining -= cur_len
                placed = True
                break
            cur_len -= 1

        # 길이 1도 더 이상 배치 불가하면 종료한다.
        if not placed:
            break

    # 마스킹 적용
    block_positions = np.where(occupied)[0]
    if len(block_positions) > 0:
        idx_masked = df.index[block_positions]
        for col in df.columns:
            masked_data.loc[idx_masked, col] = np.nan
            mask_matrix.loc[idx_masked, col] = 0.0

    actual_rate = occupied.sum() / max(n, 1)
    print(
        f"  [Block loss] block_len={block_len}pts, "
        f"목표={loss_rate*100:.0f}%, 실제={actual_rate*100:.1f}%"
    )
    return masked_data, mask_matrix, ground_truth


def apply_masking(
    df: pd.DataFrame,
    mask_type: str = "individual",
    loss_rate: float = 0.30,
    freq: str = "1H",
    block_hours: int = 48,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    통합 마스킹 인터페이스.

    mask_type: 'individual' | 'block'
    """
    if mask_type == "individual":
        return create_random_individual_mask(df, loss_rate, random_seed)
    elif mask_type == "block":
        return create_block_loss_mask(df, loss_rate, freq, block_hours, random_seed)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}. Use 'individual' or 'block'.")

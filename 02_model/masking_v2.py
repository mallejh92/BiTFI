"""
masking_v2.py
논문용 결측 시나리오 생성기

Scenario A — Individual variable missing
    하나의 변수만 gap_length_h 구간 결측. 나머지는 관측 가능 covariate.
    → CAFI가 다른 변수 정보를 활용하는 효과를 직접 평가.

Scenario B — Partial multivariate missing
    실내 센서 3개(Tin, RH, CO2)가 동시 결측. 외부(Tout, Rad)는 관측 유지.
    → 실제 데이터 로거 일부 오류 상황 모사.

Scenario C — Full multivariate block missing
    5개 변수 전체 동시 결측.
    → 통신 장애/데이터 로거 전체 오류 상황. CAFI는 iterative self-refinement 모드.

사용법:
    from masking_v2 import create_gap_masks, SCENARIO_CONFIGS, MaskResult
    masks = create_gap_masks(test_norm, orig_mask, 'A', 24, ['Tin'], n_repeats=10)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── 변수 그룹 정의 ──────────────────────────────────────────────────────────
TARGET_VARS = ["Tin", "Tout", "RH", "CO2", "Rad"]

# Scenario별 마스킹할 변수 조합 목록
# A: 변수 하나씩 (5개 combo)
# B: 실내 센서 3개 동시
# C: 전체 5개 동시
SCENARIO_CONFIGS: dict[str, list[list[str]]] = {
    "A": [[v] for v in TARGET_VARS],
    "B": [["Tin", "RH", "CO2"]],
    "C": [TARGET_VARS],
}

# 논문 표준 gap length (단위: 시간)
GAP_LENGTHS_H: list[int] = [6, 12, 24, 72, 168]

# 기본 파라미터
CONTEXT_LEN_H: int = 720    # 30일 = 720시간
N_REPEATS: int = 10
RANDOM_SEED: int = 42


# ── 결과 컨테이너 ────────────────────────────────────────────────────────────

@dataclass
class MaskResult:
    """단일 반복 실험의 마스킹 결과."""
    masked_data: pd.DataFrame       # 인위 + 원본 결측 모두 NaN
    artificial_mask: pd.DataFrame   # 0=인위 결측, 1=나머지
    effective_mask: pd.DataFrame    # 0=원본결측OR인위결측, 1=관측
    ground_truth: pd.DataFrame      # 원본 test data (역정규화 평가용)
    eval_mask: pd.DataFrame         # True = 인위결측 & 원본유효 → 평가 위치
    masked_vars: list[str]          # 이번 조합에서 가려진 변수
    gap_start_idx: int              # 갭 시작 인덱스 (test_data 기준)
    gap_end_idx: int                # 갭 끝 인덱스 (inclusive)
    repeat: int                     # 반복 번호 (0-based)
    scenario: str                   # 'A' | 'B' | 'C'
    gap_length_h: int
    context_len_h: int = CONTEXT_LEN_H


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _find_valid_gap_starts(
    original_valid_mask: pd.DataFrame,
    masked_vars: list[str],
    gap_len: int,
    ctx_len: int,
) -> np.ndarray:
    """
    갭 배치 가능한 시작 인덱스 목록을 반환한다.

    조건:
      1. start >= ctx_len  (context window 확보)
      2. gap 구간 전체에서 masked_vars 모두 original_valid_mask == 1
         (원본 결측이 포함된 구간은 평가 불가)
    """
    n = len(original_valid_mask)

    # 위치별로 masked_vars가 모두 유효한지 계산 (uint8로 합산 가능하게)
    all_valid = np.ones(n, dtype=np.uint8)
    for var in masked_vars:
        if var not in original_valid_mask.columns:
            return np.array([], dtype=np.intp)
        all_valid &= (original_valid_mask[var].values == 1).astype(np.uint8)

    # rolling sum (conv[i] = sum of all_valid[i:i+gap_len])
    conv = np.convolve(all_valid, np.ones(gap_len, dtype=np.uint8), mode="valid")

    max_start = n - gap_len  # 갭이 배열 밖으로 나가지 않아야 함
    if max_start < ctx_len:
        return np.array([], dtype=np.intp)

    candidate = np.arange(min(len(conv), max_start + 1), dtype=np.intp)
    valid = candidate[
        (candidate >= ctx_len) & (conv[candidate] == gap_len)
    ]
    return valid


def _select_nonoverlapping(
    valid_starts: np.ndarray,
    gap_len: int,
    n_repeats: int,
    rng: np.random.RandomState,
) -> list[tuple[int, int]]:
    """
    valid_starts에서 겹치지 않는 갭 위치 n_repeats개를 무작위로 선택한다.
    """
    perm = rng.permutation(valid_starts)
    occupied = np.zeros(max(int(perm.max()) + gap_len + 1, 1), dtype=bool)
    selected: list[tuple[int, int]] = []

    for start in perm:
        start = int(start)
        end = start + gap_len - 1
        if occupied[start:end + 1].any():
            continue
        occupied[start:end + 1] = True
        selected.append((start, end))
        if len(selected) >= n_repeats:
            break

    selected.sort(key=lambda x: x[0])
    return selected


# ── 공개 API ─────────────────────────────────────────────────────────────────

def create_gap_masks(
    test_data: pd.DataFrame,
    original_valid_mask: pd.DataFrame,
    scenario: str,
    gap_length_h: int,
    masked_vars: list[str],
    context_len_h: int = CONTEXT_LEN_H,
    n_repeats: int = N_REPEATS,
    random_seed: int = RANDOM_SEED,
) -> list[MaskResult]:
    """
    하나의 (scenario, gap_length_h, masked_vars) 조합에 대해
    최대 n_repeats개의 비중첩 갭 마스크를 생성한다.

    Parameters
    ----------
    test_data            : 정규화된 test DataFrame (NaN = 원본 결측)
    original_valid_mask  : 1=원본 관측, 0=원본 결측 (test_data와 같은 shape)
    scenario             : 'A' | 'B' | 'C'
    gap_length_h         : 갭 길이(시간). 1H freq 가정.
    masked_vars          : 이 조합에서 가릴 변수 목록
    context_len_h        : 갭 앞에 필요한 최소 history 길이(시간)
    n_repeats            : 목표 반복 수 (가능한 만큼, 최대 n_repeats)
    random_seed          : 재현성

    Returns
    -------
    list[MaskResult]   # 길이 ≤ n_repeats
    """
    gap_len = gap_length_h      # 1H freq → 1포인트 = 1시간
    ctx_len = context_len_h

    rng = np.random.RandomState(random_seed)

    valid_starts = _find_valid_gap_starts(
        original_valid_mask, masked_vars, gap_len, ctx_len
    )
    if len(valid_starts) == 0:
        return []

    gaps = _select_nonoverlapping(valid_starts, gap_len, n_repeats, rng)

    results: list[MaskResult] = []
    for repeat_idx, (gs, ge) in enumerate(gaps):
        # ── artificial_mask: 0=인위결측, 1=나머지 ──
        art_mask = pd.DataFrame(
            np.ones(test_data.shape, dtype=np.float32),
            index=test_data.index,
            columns=test_data.columns,
        )
        for var in masked_vars:
            if var in art_mask.columns:
                col_pos = art_mask.columns.get_loc(var)
                art_mask.iloc[gs:ge + 1, col_pos] = 0.0

        # ── masked_data: 인위 결측 위치에 NaN 추가 ──
        masked = test_data.copy()
        for var in masked_vars:
            if var in masked.columns:
                col_pos = masked.columns.get_loc(var)
                masked.iloc[gs:ge + 1, col_pos] = np.nan

        # ── effective_mask = original_valid_mask * artificial_mask ──
        # 0이면 모델 입력에서 결측으로 처리해야 함
        eff_mask = (original_valid_mask * art_mask).astype(np.float32)

        # ── eval_mask: 평가 위치 (artificial=0 AND original_valid=1) ──
        eval_mask = (art_mask == 0) & (original_valid_mask == 1)

        results.append(MaskResult(
            masked_data=masked,
            artificial_mask=art_mask,
            effective_mask=eff_mask,
            ground_truth=test_data.copy(),
            eval_mask=eval_mask,
            masked_vars=list(masked_vars),
            gap_start_idx=gs,
            gap_end_idx=ge,
            repeat=repeat_idx,
            scenario=scenario,
            gap_length_h=gap_length_h,
            context_len_h=context_len_h,
        ))

    return results


def create_all_masks_for_scenario(
    test_data: pd.DataFrame,
    original_valid_mask: pd.DataFrame,
    scenario: str,
    gap_lengths_h: list[int] = GAP_LENGTHS_H,
    context_len_h: int = CONTEXT_LEN_H,
    n_repeats: int = N_REPEATS,
    random_seed: int = RANDOM_SEED,
) -> dict[tuple[str, int, str], list[MaskResult]]:
    """
    scenario 내 모든 (masked_vars, gap_length_h) 조합의 마스크를 반환한다.

    Returns
    -------
    dict: (scenario, gap_length_h, masked_vars_str) → list[MaskResult]
    """
    var_combos = SCENARIO_CONFIGS.get(scenario, [])
    results: dict[tuple[str, int, str], list[MaskResult]] = {}

    for masked_vars in var_combos:
        vars_key = ",".join(masked_vars)
        for gap_h in gap_lengths_h:
            masks = create_gap_masks(
                test_data=test_data,
                original_valid_mask=original_valid_mask,
                scenario=scenario,
                gap_length_h=gap_h,
                masked_vars=masked_vars,
                context_len_h=context_len_h,
                n_repeats=n_repeats,
                random_seed=random_seed + hash(f"{scenario}{gap_h}{vars_key}") % 1000,
            )
            key = (scenario, gap_h, vars_key)
            results[key] = masks
            print(
                f"  [mask] scenario={scenario} gap={gap_h}h "
                f"vars={vars_key}: {len(masks)}/{n_repeats} 반복 생성"
            )

    return results

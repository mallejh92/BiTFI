"""
anomaly_detection.py
모델 잔차 기반 스파이크(spike) 탐지 핵심 함수 (이상치탐지_계획.md 구현, spike-only)

원리:
  1. 학습된 모델(CAFI/AutoGluon/Chronos2/TimesFM)로 전체 시계열을 교차 블록
     마스킹 스캔해 "관측을 보지 않은" 재구성값을 얻고, 잔차(obs-pred)를 계산.
  2. 잔차를 변수별 Rolling-MAD로 robust 표준화해 |z| > tau 인 점을 스파이크로 flag.
     (드리프트·고착은 이 온실 데이터에는 해당 없다고 판단해 탐지 대상에서 제외함.)
  3. 여러 모델의 flag를 평균해 합의(consensus) 점수를 만든다.
  4. 합성 스파이크 주입으로 탐지 정밀도/재현율과 "복원 정확도"를 함께 검증한다
     (라벨이 없으므로 — 알고 있는 위치에 알고 있는 크기로 주입 후 비교).

이미 학습된 비교 모델(checkpoint.py로 저장된)을 재사용하며, 신규 학습은 하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 1) 교차 블록 마스킹 잔차 스캔
# ──────────────────────────────────────────────────────────────────────────────

def _alternating_block_vector(n: int, block: int, phase: int) -> np.ndarray:
    """길이 n에서 block 크기 블록을 phase(0|1)에 따라 번갈아 마스킹(0)하는 벡터.

    인접 블록을 동시에 마스킹하지 않아(phase 0=짝수 블록, 1=홀수 블록) 모델이
    참조할 컨텍스트가 항상 남아있다. 두 phase를 합치면 전체 구간이 커버된다.
    """
    vec = np.ones(n, dtype=np.float32)
    for start in range(phase * block, n, 2 * block):
        vec[start:start + block] = 0.0
    return vec


def scan_residuals(
    model,
    full_data: pd.DataFrame,
    target_cols: list[str],
    block_h: int = 6,
) -> pd.DataFrame:
    """
    전체 시계열을 2-패스 교차 블록 마스킹으로 스캔해 target_cols에 대한
    모델 재구성 잔차(obs - pred)를 계산한다.

    full_data : 모델에 전달할 전체 입력 프레임. CAFI처럼 covariate를 함께
                요구하는 모델은 target_cols + covariate_cols를 모두 포함한
                프레임을 넘겨야 한다(covariate 컬럼은 마스킹하지 않음).
    target_cols : 잔차를 계산할 컬럼(보통 5개 핵심 타깃). covariate는 항상
                  관측(1)으로 유지되어 마스킹 대상에서 제외된다.

    Returns
    -------
    DataFrame (index=full_data.index, columns=target_cols)
        원본 결측 위치는 NaN(잔차 미정의).
    """
    n = len(full_data)
    all_cols = list(full_data.columns)
    recon = full_data[target_cols].copy().astype(float)

    for phase in (0, 1):
        block_vec = _alternating_block_vector(n, block_h, phase)
        mask_df = pd.DataFrame(1.0, index=full_data.index, columns=all_cols)
        for c in target_cols:
            mask_df[c] = block_vec

        masked_df = full_data.copy()
        masked_df[mask_df == 0] = np.nan

        imputed = model.impute(masked_df, mask_df)

        fill_pos = mask_df[target_cols] == 0
        recon[target_cols] = recon[target_cols].where(~fill_pos, imputed[target_cols])

    residual = full_data[target_cols].astype(float) - recon[target_cols]
    residual[full_data[target_cols].isna()] = np.nan
    return residual


# ──────────────────────────────────────────────────────────────────────────────
# 2) 잔차 robust 표준화 (스파이크)
# ──────────────────────────────────────────────────────────────────────────────

def mad_zscore(residual: pd.Series) -> pd.Series:
    """잔차의 MAD 기반 robust z-score. 결측·평탄(MAD=0) 위치는 NaN."""
    valid = residual.dropna()
    if len(valid) < 5:
        return pd.Series(np.nan, index=residual.index)
    med = valid.median()
    mad = (valid - med).abs().median() * 1.4826
    if mad < 1e-9:
        return pd.Series(np.nan, index=residual.index)
    return (residual - med).abs() / mad


def spike_flag(z: pd.Series, tau: float = 3.5) -> pd.Series:
    """|z| > tau 인 위치를 스파이크로 flag(bool)."""
    return (z > tau).fillna(False)


# ──────────────────────────────────────────────────────────────────────────────
# 3) 다중모델 합의
# ──────────────────────────────────────────────────────────────────────────────

def consensus_score(flags: dict[str, pd.Series]) -> pd.Series:
    """모델별 flag(bool Series, 동일 index)를 평균해 합의 점수(0~1)를 만든다."""
    df = pd.DataFrame(flags)
    return df.mean(axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# 4) 합성 스파이크 주입 + 탐지/복원 검증
# ──────────────────────────────────────────────────────────────────────────────

def inject_synthetic_spikes(
    clean: pd.Series,
    valid_mask: pd.Series,
    n_inject: int = 20,
    seed: int = 42,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    원본 결측이 없는(valid_mask=True) 구간의 단일 지점에 합성 스파이크를
    무작위 주입한다(±4~8 표준편차 급등락, 단일 지점).

    Returns
    -------
    (주입된 series, injected 위치 DataFrame[start, end, true_value])
        start==end(스파이크는 항상 1포인트). true_value는 주입 전 원래 값
        (복원 정확도 평가에 사용).
    """
    rng = np.random.RandomState(seed)
    s = clean.copy()
    valid = valid_mask.values.astype(bool)
    n = len(s)
    std = float(s.std()) or 1.0
    records: list[dict] = []
    attempts = 0
    max_attempts = n_inject * 30

    while len(records) < n_inject and attempts < max_attempts:
        attempts += 1
        t = int(rng.randint(0, n))
        if not valid[t] or any(t == rec["start"] for rec in records):
            continue
        true_value = float(s.iloc[t])
        s.iloc[t] = true_value + rng.choice([-1, 1]) * rng.uniform(4, 8) * std
        records.append({"start": t, "end": t, "true_value": true_value})

    return s, pd.DataFrame(records, columns=["start", "end", "true_value"])


def evaluate_detection(flag: pd.Series, injected: pd.DataFrame, n: int) -> dict:
    """주입 위치(injected) 대비 flag의 precision/recall/F1을 계산한다."""
    truth = np.zeros(n, dtype=bool)
    for _, row in injected.iterrows():
        truth[int(row["start"]):int(row["end"]) + 1] = True
    pred = flag.values.astype(bool)
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision and recall and (precision + recall) > 0)
        else np.nan
    )
    return {"precision": precision, "recall": recall, "f1": f1, "n_injected": len(injected)}


def restoration_error(restored: pd.Series, injected: pd.DataFrame) -> dict:
    """복원된 series가 주입 위치에서 원래 진짜 값(true_value)에 얼마나 가까운지 측정."""
    if injected.empty:
        return {"mae": np.nan, "n": 0}
    idx = injected["start"].astype(int).values
    pred = restored.iloc[idx].values.astype(float)
    true = injected["true_value"].values.astype(float)
    mae = float(np.mean(np.abs(pred - true)))
    return {"mae": mae, "n": len(injected)}


# ──────────────────────────────────────────────────────────────────────────────
# 5) 결측 표시 후 동일 보간 파이프라인으로 복원
# ──────────────────────────────────────────────────────────────────────────────

def restore_flagged(
    model,
    full_data: pd.DataFrame,
    target_cols: list[str],
    flagged_mask: pd.DataFrame,
) -> pd.DataFrame:
    """
    flagged_mask(True=이상치로 판정된 위치, target_cols 컬럼만)를 결측으로
    표시한 뒤, 같은 모델의 impute()로 복원한다(개선사항.md #2 후속 처리).
    """
    all_cols = list(full_data.columns)
    masked = full_data.copy()
    mask_df = pd.DataFrame(1.0, index=full_data.index, columns=all_cols)
    for c in target_cols:
        if c in flagged_mask.columns:
            flagged = flagged_mask[c].astype(bool)
            masked.loc[flagged, c] = np.nan
            mask_df.loc[flagged, c] = 0.0
    imputed = model.impute(masked, mask_df)
    result = full_data.copy()
    result[target_cols] = imputed[target_cols]
    return result

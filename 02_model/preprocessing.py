"""
preprocessing.py
온실 환경 시계열 데이터 전처리 모듈
- xlsx/csv 자동 감지 및 로드
- 아웃라이어 제거 → 단기 결측 선형 보간 → MinMax 정규화
- 핵심 5개 변수 선택 및 rename
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# 핵심 5개 변수 매핑 (한글 컬럼명 → 영문 약어)
TARGET_COLS = {
    "내부-내부온도": "Tin",
    "외부-외부온도": "Tout",
    "내부-내부습도": "RH",
    "내부-내부CO2": "CO2",
    "외부-외부일사량": "Rad",
}

# 보조 covariate 매핑 (한글 컬럼명 → 영문 약어).
# CAFI covariate 확장(개선사항.md #5)에 사용. 타깃이 아니라 보조 입력이며,
# 파일에 컬럼이 없으면 자동으로 제외된다(coverage가 온실마다 다름).
# NOTE: 원천 데이터(01_data/*.xlsx)는 .gitignore로 제외되어 실제 한글 컬럼명을
#       이 저장소에서 직접 확인할 수 없다. 아래 값은 기존 명명 규칙
#       ("외부-외부...", "내부-내부...")을 따른 best-guess이며, 실제 컬럼명이
#       다르면 이 dict만 수정하면 된다(없으면 graceful하게 무시됨).
COVARIATE_COLS = {
    "외부-외부풍속": "WindSpeed",   # 외부 풍속
    "내부-내부일사량": "RadIn",      # 내부 일사 (외부 일사 Rad와 별개 센서)
}

# 물리적 유효 범위
VALID_RANGES = {
    "Tin":  (-30, 70),
    "Tout": (-30, 70),
    "RH":   (0, 100),
    "CO2":  (0, 5000),
    "Rad":  (0, 2000),
}

# covariate 물리 범위 (이상치 → NaN 용). 등록되지 않은 covariate는 범위검사 생략.
COVARIATE_VALID_RANGES = {
    "WindSpeed": (0, 60),     # m/s
    "RadIn":     (0, 2000),   # W/m^2
}

# 단기 결측 선형 보간 한계 (포인트 수)
SHORT_GAP_LIMIT = 3

# NOTE(개선사항.md #2): Rolling-MAD(Hampel) 기반 통계적 이상치 탐지를 시도했으나,
# flag된 지점들을 실측 검증한 결과 대부분 "그날만 날씨가 달라 같은 시간대
# 평소와 다른" 정상적인 환경 변동(부드러운 추세)이었고, 센서 글리치 특유의
# 스파이크-후-즉시복귀 패턴이 아니었다. 환경 데이터는 날씨에 따라 자연스럽게
# 변하므로 통계적 이상치 탐지는 적용하지 않고, 물리적으로 불가능한 값(범위
# 밖)만 제거한다.


def _detect_freq(index: pd.DatetimeIndex) -> str:
    """시간 간격 자동 감지 → pandas freq 문자열 반환"""
    if len(index) < 2:
        return "1H"
    diffs = index.to_series().diff().dropna()
    median_diff = diffs.median()
    minutes = median_diff.total_seconds() / 60
    if minutes <= 1.5:
        return "1min"
    elif minutes <= 6:
        return "5min"
    elif minutes <= 11:
        return "10min"
    else:
        return "1H"


def _load_file(filepath: str | Path) -> pd.DataFrame:
    """xlsx 또는 csv 파일을 로드하고 DatetimeIndex를 설정한다."""
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, dtype=str)
    elif ext == ".csv":
        df = pd.read_csv(filepath, encoding="utf-8-sig", dtype=str)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")

    # 날짜 컬럼 찾기
    date_col = None
    for c in df.columns:
        if "수집일" in c or "date" in c.lower() or "time" in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    # datetime 파싱: "2024-08-02 000000" 형식 우선, 실패 시 자동
    def _parse_dt(s):
        s = str(s).strip()
        # "2024-08-02 000000" → "2024-08-02 00:00:00"
        if len(s) >= 15 and s[10] == " " and ":" not in s[11:]:
            s = s[:11] + s[11:13] + ":" + s[13:15] + ":" + s[15:17]
        return pd.to_datetime(s, errors="coerce")

    df[date_col] = df[date_col].apply(_parse_dt)
    if df[date_col].isna().any():
        raise ValueError(f"Unparseable timestamps in {filepath}")
    df = df.set_index(date_col).sort_index()
    df.index.name = "datetime"

    # 숫자형 변환
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return regularize_hourly(df)


def regularize_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Restore missing hourly timestamps without filling any sensor value.

    Source timestamps are local civil times (Korea, no daylight-saving change).
    Duplicate or off-hour timestamps require explicit source correction.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        raise ValueError("A nonempty DatetimeIndex is required")
    if df.index.hasnans or df.index.has_duplicates:
        raise ValueError("Missing or duplicate timestamps")
    frame = df.sort_index()
    if not frame.index.equals(frame.index.floor("h")):
        raise ValueError("Off-hour timestamps cannot be treated as hourly records")
    grid = pd.date_range(frame.index[0], frame.index[-1], freq="h", name=frame.index.name)
    result = frame.reindex(grid)
    result.attrs.update(source_rows=len(frame), inserted_hourly_rows=len(grid)-len(frame),
                        time_grid="hourly; unrecorded timestamps retained as NaN")
    assert_hourly(result.index)
    return result


def assert_hourly(index: pd.DatetimeIndex) -> None:
    # Pure array checks avoid pandas' lazy shared index-engine cache in workers.
    stamps = index.asi8
    if np.any(stamps == np.iinfo(np.int64).min):
        raise AssertionError("Missing timestamp")
    if len(stamps) > 1 and not np.all(np.diff(stamps) == pd.Timedelta(hours=1).value):
        raise AssertionError("Each adjacent position must be one elapsed hour")


def _select_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """파일에 존재하는 covariate 컬럼만 골라 영문으로 rename. 없으면 빈 DataFrame."""
    present = {kr: en for kr, en in COVARIATE_COLS.items() if kr in df.columns}
    if not present:
        return pd.DataFrame(index=df.index)
    return df[list(present.keys())].rename(columns=present)


def _select_and_rename(
    df: pd.DataFrame, include_covariates: bool = False
) -> pd.DataFrame | None:
    """
    핵심 5개 변수가 모두 있을 때만 선택하고 영문으로 rename.
    include_covariates=True이면 존재하는 covariate 컬럼을 뒤에 덧붙인다(없으면 무시).
    """
    # 핵심 컬럼 5개 중 하나라도 누락되면 파일 전체를 skip한다.
    missing = [k for k in TARGET_COLS if k not in df.columns]
    if missing:
        print(f"  핵심 변수 누락({len(missing)}개): {missing} → skip")
        return None
    sub = df[list(TARGET_COLS.keys())].rename(columns=TARGET_COLS)
    if include_covariates:
        cov = _select_covariates(df)
        if not cov.empty:
            sub = pd.concat([sub, cov], axis=1)
            print(f"  covariate 사용: {list(cov.columns)}")
    return sub


def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """물리적으로 불가능한 값을 NaN으로 대체 (타깃 + 등록된 covariate)."""
    df = df.copy()
    ranges = {**VALID_RANGES, **COVARIATE_VALID_RANGES}
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            n = mask.sum()
            if n > 0:
                print(f"  [{col}] 아웃라이어 {n}개 제거 (range: {lo}~{hi})")
            df.loc[mask, col] = np.nan
    return df


def _interpolate_short_gaps(df: pd.DataFrame, limit: int = SHORT_GAP_LIMIT) -> pd.DataFrame:
    """연속 결측이 limit 포인트 이하인 구간만 선형 보간"""
    return df.interpolate(method="linear", limit=limit, limit_direction="both")


def preprocess_file(
    filepath: str | Path,
    scaler_dir: str | Path | None = None,
    fit_scaler: bool = True,
    existing_scaler: dict | None = None,
    include_covariates: bool = False,
) -> dict | None:
    """
    단일 파일 전처리.

    Parameters
    ----------
    include_covariates : True이면 존재하는 covariate 컬럼(COVARIATE_COLS)을
                         data/data_raw에 함께 포함한다(없으면 무시). 기본 False로
                         기존 러너 동작과 완전 하위호환.

    Returns
    -------
    dict with keys:
        'name': 파일 스템
        'data': 정규화된 DataFrame (5개 변수 + (옵션)covariate)
        'data_raw': 정규화 전 DataFrame
        'scaler': dict of {col: MinMaxScaler}
        'freq': 감지된 freq 문자열
        'missing_rate': 변수별 최종 결측 비율
        'covariates': 포함된 covariate 영문 컬럼 목록 (없으면 [])
    """
    filepath = Path(filepath)
    print(f"\n{'='*60}")
    print(f"로딩: {filepath.name}")

    try:
        df_full = _load_file(filepath)
    except Exception as e:
        print(f"  로드 실패: {e}")
        return None

    df = _select_and_rename(df_full, include_covariates=include_covariates)
    if df is None:
        print("  핵심 변수 컬럼 없음 → skip")
        return None

    covariate_cols = [c for c in df.columns if c in COVARIATE_COLS.values()]

    freq = _detect_freq(df.index)
    print(f"  감지된 시간 간격: {freq} | 행 수: {len(df)}")

    # 1) 물리 범위 이상치 → NaN (불가능한 값만 제거. 통계적 이상치 탐지는
    #    실측 검증 결과 정상적인 날씨발 변동을 오탐해 적용하지 않는다 — 개선사항.md #2)
    df = _remove_outliers(df)

    # 2) 단기 결측 선형 보간
    # NOTE(개선사항.md #1/L1): 이 보간은 train/test 분리 *이전*에 수행되므로,
    #   단일-온실 시간순 분할(train_test_split)에서는 80/20 경계의 최대
    #   SHORT_GAP_LIMIT 포인트가 양방향으로 채워져 경미한 누수가 생길 수 있다.
    #   교차-온실 분할(data_split.build_cross_greenhouse_split)에서는 온실 경계가
    #   곧 train/test 경계이므로 이 누수가 원천 소멸한다.
    df = _interpolate_short_gaps(df)

    # 결측 비율 출력
    missing_rate = df.isnull().mean()
    print("  최종 결측 비율:")
    for col, rate in missing_rate.items():
        print(f"    {col}: {rate*100:.1f}%")

    data_raw = df.copy()

    # MinMax 정규화
    scalers = {}
    df_norm = df.copy()

    if fit_scaler:
        for col in df.columns:
            scaler = MinMaxScaler()
            valid = df[col].dropna().values.reshape(-1, 1)
            if len(valid) == 0:
                scalers[col] = None
                continue
            scaler.fit(valid)
            df_norm[col] = df[col].copy()
            not_null = df[col].notna()
            df_norm.loc[not_null, col] = scaler.transform(
                df.loc[not_null, col].values.reshape(-1, 1)
            ).flatten()
            scalers[col] = scaler
    else:
        scalers = existing_scaler or {}
        for col in df.columns:
            scaler = scalers.get(col)
            if scaler is None:
                continue
            not_null = df[col].notna()
            df_norm.loc[not_null, col] = scaler.transform(
                df.loc[not_null, col].values.reshape(-1, 1)
            ).flatten()

    # 스케일러 저장
    if scaler_dir is not None and fit_scaler:
        scaler_dir = Path(scaler_dir)
        scaler_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = scaler_dir / f"{filepath.stem}_scalers.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scalers, f)
        print(f"  스케일러 저장: {scaler_path}")

    return {
        "name": filepath.stem,
        "data": df_norm,
        "data_raw": data_raw,
        "scaler": scalers,
        "freq": freq,
        "missing_rate": missing_rate.to_dict(),
        "covariates": covariate_cols,
    }


def preprocess_directory(
    data_dir: str | Path,
    scaler_dir: str | Path | None = None,
    extensions: tuple = (".xlsx", ".xls", ".csv"),
) -> list[dict]:
    """
    data_dir 내 모든 데이터 파일을 순차 전처리.

    Returns
    -------
    list of dicts (preprocess_file 반환값)
    """
    data_dir = Path(data_dir)
    files = sorted(
        [f for f in data_dir.iterdir() if f.suffix.lower() in extensions]
    )
    if not files:
        print(f"'{data_dir}' 에 데이터 파일 없음")
        return []

    results = []
    for fp in files:
        result = preprocess_file(fp, scaler_dir=scaler_dir)
        if result is not None:
            results.append(result)

    print(f"\n전처리 완료: {len(results)}/{len(files)} 파일")
    return results


def train_test_split(result: dict, test_ratio: float = 0.2) -> tuple[dict, dict]:
    """
    단일 온실 데이터를 시간 순서로 train/test 분리.
    마지막 test_ratio 비율을 test set으로 사용.
    """
    df = result["data"]
    df_raw = result["data_raw"]
    n = len(df)
    split_idx = int(n * (1 - test_ratio))

    train = result.copy()
    train["data"] = df.iloc[:split_idx]
    train["data_raw"] = df_raw.iloc[:split_idx]

    test = result.copy()
    test["data"] = df.iloc[split_idx:]
    test["data_raw"] = df_raw.iloc[split_idx:]

    return train, test


def fit_scalers_from_split(
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    train 원본 데이터만으로 MinMaxScaler를 fit하고 train/test를 정규화한다.
    test 분포를 scaler에 노출하지 않아 데이터 누수를 방지한다.
    """
    scalers: dict = {}
    df_train_norm = df_train_raw.copy().astype(float)
    df_test_norm  = df_test_raw.copy().astype(float)

    for col in df_train_raw.columns:
        valid = df_train_raw[col].dropna().values.reshape(-1, 1)
        if len(valid) == 0:
            scalers[col] = None
            continue
        scaler = MinMaxScaler()
        scaler.fit(valid)
        scalers[col] = scaler
        for df_raw, df_norm in [
            (df_train_raw, df_train_norm),
            (df_test_raw,  df_test_norm),
        ]:
            not_null = df_raw[col].notna()
            if not_null.any():
                df_norm.loc[not_null, col] = scaler.transform(
                    df_raw.loc[not_null, col].values.reshape(-1, 1)
                ).flatten()

    return scalers, df_train_norm, df_test_norm

"""
data_split.py
교차-온실(cross-greenhouse) 데이터 분할 (개선사항.md #4, #5)

기존 온실별 80/20 시간순 분할은 데이터 기간(2024-08~2025-06) 탓에 Test가 항상
봄(4~6월)만 되어 계절 범용성을 평가하지 못했다. 본 모듈은 **온실 경계**를 train/test
경계로 삼아:
  - Test 온실 = 핵심 5변수 완비 ∩ (여름+겨울 모두 포함) ∩ covariate 2종 보유
    → Test가 1년 전체를 담으므로 여름/겨울·오전/오후 층화 분석이 가능(=#4 해결).
  - Train 온실 = 핵심변수 일부 결손 온실 + 완비 중 Test 미선정분
    → 결손 온실을 학습에 편입해 데이터 낭비를 줄임(=#5).
  - 온실 경계=분할 경계라 분리 전 보간(L1) 누수가 원천 소멸(=#1).

핵심변수가 하나도 없는 파일(PF_0001536 등)은 학습/평가 모두에서 제외한다.

사용:
    from data_split import build_cross_greenhouse_split
    split = build_cross_greenhouse_split("../01_data", n_test=10, seed=42)
    # split["train"], split["test"] = 파일 경로 목록
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np

from preprocessing import _load_file, TARGET_COLS, COVARIATE_COLS

# 계절 정의 (월 기준). 분류·시각화에서 공통 사용.
SUMMER_MONTHS = {6, 7, 8}
WINTER_MONTHS = {12, 1, 2}


def _scan_file(path: Path) -> dict | None:
    """
    단일 파일의 변수 완비성·계절 커버리지·covariate 보유 여부를 스캔한다.

    Returns None이면 핵심변수 전무(학습/평가 제외 대상) 또는 로드 실패.
    """
    try:
        df = _load_file(path)
    except Exception as e:
        print(f"  [scan] {path.name} 로드 실패: {e}")
        return None

    have_targets = [k for k in TARGET_COLS if k in df.columns]
    if len(have_targets) == 0:
        # 핵심변수가 하나도 없으면 글로벌 학습에도 기여 못 함 → 완전 제외.
        return None

    have_cov = [k for k in COVARIATE_COLS if k in df.columns]
    months = set(int(m) for m in np.unique(df.index.month)) if len(df) else set()

    return {
        "name": path.stem,
        "path": str(path),
        "n_targets": len(have_targets),
        "complete": len(have_targets) == len(TARGET_COLS),
        "n_covariates": len(have_cov),
        "has_all_covariates": len(have_cov) == len(COVARIATE_COLS),
        "months": sorted(months),
        "has_summer": bool(months & SUMMER_MONTHS),
        "has_winter": bool(months & WINTER_MONTHS),
        "n_rows": int(len(df)),
    }


def build_cross_greenhouse_split(
    data_dir: str | Path,
    n_test: int = 10,
    seed: int = 42,
    result_dir: str | Path | None = None,
    require_covariates: bool = True,
) -> dict:
    """
    교차-온실 분할을 구성하고 split.json으로 저장한다.

    Parameters
    ----------
    data_dir           : PF_*.xlsx 들이 있는 디렉터리
    n_test             : Test 홀드아웃 온실 수 (Test 풀에서 무작위 추출)
    seed               : 재현성 시드
    result_dir         : split.json 저장 위치. None이면 ../03_result/comparison.
    require_covariates : True면 Test 풀이 covariate 2종 보유 온실로 한정(CAFI 평가용).

    Returns
    -------
    dict: {"train": [paths], "test": [paths], "meta": {...}}
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("PF_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"'{data_dir}'에 PF_*.xlsx 파일이 없습니다.")

    print(f"[data_split] {len(files)}개 파일 스캔 중...")
    scans = [s for s in (_scan_file(fp) for fp in files) if s is not None]
    print(f"[data_split] 유효(핵심변수 ≥1) 온실: {len(scans)}개")

    complete = [s for s in scans if s["complete"]]
    deficient = [s for s in scans if not s["complete"]]

    # Test 풀: 완비 ∩ 여름·겨울 모두 포함 ∩ (옵션)covariate 2종 보유
    def _eligible(s: dict) -> bool:
        ok = s["complete"] and s["has_summer"] and s["has_winter"]
        if require_covariates:
            ok = ok and s["has_all_covariates"]
        return ok

    test_pool = [s for s in complete if _eligible(s)]
    print(
        f"[data_split] 완비 {len(complete)}개 | 결손 {len(deficient)}개 | "
        f"Test 풀(완비∩전계절{'∩covariate' if require_covariates else ''}) {len(test_pool)}개"
    )

    if len(test_pool) < n_test:
        print(
            f"  ⚠ Test 풀({len(test_pool)})이 n_test({n_test})보다 작음 → "
            f"가능한 {len(test_pool)}개만 Test로 사용."
        )
        n_test = len(test_pool)

    rng = np.random.RandomState(seed)
    test_names = set(
        rng.choice([s["name"] for s in test_pool], size=n_test, replace=False).tolist()
    ) if n_test > 0 else set()

    test = [s for s in scans if s["name"] in test_names]
    # Train = 나머지 전부(결손 + 완비 중 Test 미선정). 핵심변수 전무 파일은 이미 제외됨.
    train = [s for s in scans if s["name"] not in test_names]

    split = {
        "train": [s["path"] for s in train],
        "test": [s["path"] for s in test],
        "meta": {
            "data_dir": str(data_dir),
            "seed": seed,
            "n_test": n_test,
            "require_covariates": require_covariates,
            "n_files_scanned": len(files),
            "n_valid": len(scans),
            "n_complete": len(complete),
            "n_deficient": len(deficient),
            "n_test_pool": len(test_pool),
            "test_greenhouses": sorted(test_names),
            "train_greenhouses": sorted(s["name"] for s in train),
            "scans": {s["name"]: s for s in scans},
        },
    }

    if result_dir is None:
        result_dir = _THIS_DIR / ".." / "03_result" / "comparison"
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    split_path = result_dir / "split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    print(f"[data_split] 분할 저장: {split_path}")
    print(f"  Train {len(train)}개 / Test {len(test)}개")

    return split


def load_split(result_dir: str | Path | None = None) -> dict:
    """저장된 split.json을 로드한다(재현성 고정 재사용)."""
    if result_dir is None:
        result_dir = _THIS_DIR / ".." / "03_result" / "comparison"
    split_path = Path(result_dir) / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"split.json 없음: {split_path}. 먼저 build_cross_greenhouse_split 실행.")
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="교차-온실 데이터 분할 생성")
    p.add_argument("--data-dir", type=str, default="../01_data")
    p.add_argument("--n-test", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--result-dir", type=str, default="../03_result/comparison")
    p.add_argument("--no-covariate-filter", action="store_true",
                   help="Test 풀에서 covariate 2종 보유 조건을 제거")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = (_THIS_DIR / data_dir).resolve()
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = (_THIS_DIR / result_dir).resolve()

    build_cross_greenhouse_split(
        data_dir=data_dir,
        n_test=args.n_test,
        seed=args.seed,
        result_dir=result_dir,
        require_covariates=not args.no_covariate_filter,
    )

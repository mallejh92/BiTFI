"""
data_split.py
Cross-greenhouse train/test split.

The split separates greenhouses rather than time periods. The released split is
frozen in ``01_data/split.json``: ten test greenhouses and 33 remaining sites
(24 with all five targets form the training pool). Re-sampling the test sites
would change every reported result, so this module no longer draws a new split.
It resolves the frozen filenames against the data directory, checks that every
test site has all five target columns and timestamps spanning both June--August
and December--February, and writes the resolved split to
``result_dir/split.json``. The original 20-site test pool also required two
auxiliary sensor columns that no model uses; ``meta.selection_rule`` in the
frozen split records the full rule.

Files without any target column are excluded from training and evaluation.

Usage:
    from data_split import build_cross_greenhouse_split
    split = build_cross_greenhouse_split("../01_data", n_test=10, seed=42)
    # split["train"], split["test"] = lists of file paths
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np

from preprocessing import _load_file, TARGET_COLS

# Season definitions (months), shared by classification and plotting.
SUMMER_MONTHS = {6, 7, 8}
WINTER_MONTHS = {12, 1, 2}

FROZEN_SPLIT_PATH = _THIS_DIR / ".." / "01_data" / "split.json"


def _scan_file(path: Path) -> dict | None:
    """
    Scan one file for target completeness and seasonal timestamp coverage.

    Returns None when the file cannot be loaded or has no target column.
    """
    try:
        df = _load_file(path)
    except Exception as e:
        print(f"  [scan] failed to load {path.name}: {e}")
        return None

    have_targets = [k for k in TARGET_COLS if k in df.columns]
    if len(have_targets) == 0:
        # A file without any target cannot contribute to global training.
        return None

    months = set(int(m) for m in np.unique(df.index.month)) if len(df) else set()

    return {
        "name": path.stem,
        "path": str(path),
        "n_targets": len(have_targets),
        "complete": len(have_targets) == len(TARGET_COLS),
        "months": sorted(months),
        "has_summer": bool(months & SUMMER_MONTHS),
        "has_winter": bool(months & WINTER_MONTHS),
        "n_rows": int(len(df)),
    }


def _is_test_eligible(s: dict) -> bool:
    """All five targets and timestamps in both summer and winter."""
    return s["complete"] and s["has_summer"] and s["has_winter"]


def _resolve(name: str, data_dir: Path) -> Path:
    hits = list(data_dir.rglob(Path(name).name))
    if len(hits) != 1:
        raise ValueError(f"{name}: expected one matching input file, got {len(hits)}")
    return hits[0].resolve()


def build_cross_greenhouse_split(
    data_dir: str | Path,
    n_test: int = 10,
    seed: int = 42,
    result_dir: str | Path | None = None,
    frozen_split: str | Path | None = None,
) -> dict:
    """
    Resolve the frozen cross-greenhouse split and save it as split.json.

    Parameters
    ----------
    data_dir     : directory containing the PF_*.xlsx files
    n_test       : expected number of test greenhouses (checked against the frozen split)
    seed         : expected sampling seed (checked against the frozen split)
    result_dir   : output location of split.json; defaults to ../03_result/comparison
    frozen_split : released split file; defaults to ../01_data/split.json

    Returns
    -------
    dict: {"train": [paths], "test": [paths], "meta": {...}}
    """
    data_dir = Path(data_dir)
    frozen_split = Path(frozen_split) if frozen_split is not None else FROZEN_SPLIT_PATH
    source = json.loads(frozen_split.read_text(encoding="utf-8"))
    source_meta = source.get("meta", {})

    if source_meta.get("seed", seed) != seed or len(source["test"]) != n_test:
        raise ValueError(
            f"Requested n_test={n_test}, seed={seed} differ from the frozen split "
            f"(n_test={len(source['test'])}, seed={source_meta.get('seed')}). "
            "A different split would invalidate the reported results."
        )

    test_paths = [_resolve(n, data_dir) for n in source["test"]]
    train_paths = [_resolve(n, data_dir) for n in source["train"]]

    print(f"[data_split] checking {len(test_paths)} frozen test sites...")
    scans = {}
    for fp in test_paths + train_paths:
        s = _scan_file(fp)
        if s is not None:
            scans[s["name"]] = s
    ineligible = [p.stem for p in test_paths if not (p.stem in scans and _is_test_eligible(scans[p.stem]))]
    if ineligible:
        raise ValueError(f"Frozen test sites fail the eligibility criteria: {ineligible}")

    split = {
        "train": [str(p) for p in train_paths],
        "test": [str(p) for p in test_paths],
        "meta": {
            **source_meta,
            "data_dir": str(data_dir),
            "frozen_split": str(frozen_split),
            "n_complete_train": sum(scans[p.stem]["complete"] for p in train_paths if p.stem in scans),
            "scans": scans,
        },
    }

    if result_dir is None:
        result_dir = _THIS_DIR / ".." / "03_result" / "comparison"
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    split_path = result_dir / "split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    print(f"[data_split] saved split: {split_path}")
    print(f"  Train {len(train_paths)} / Test {len(test_paths)}")

    return split


def load_split(result_dir: str | Path | None = None) -> dict:
    """Load a saved split.json for reuse."""
    if result_dir is None:
        result_dir = _THIS_DIR / ".." / "03_result" / "comparison"
    split_path = Path(result_dir) / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"split.json not found: {split_path}. Run build_cross_greenhouse_split first.")
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Resolve the frozen cross-greenhouse split")
    p.add_argument("--data-dir", type=str, default="../01_data")
    p.add_argument("--n-test", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--result-dir", type=str, default="../03_result/comparison")
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
    )

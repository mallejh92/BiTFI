"""
checkpoint.py
모델 학습 산출물 저장/재사용 관리 (개선사항.md #5: 1회 학습 후 재사용)

체크포인트 루트 구조:
    <root>/<MODEL>/            ← 모델별 저장 디렉터리
        predictor/            ← AutoGluon predictor (디렉터리)
        cafi_state.pkl        ← CAFI Ridge 상태 (pkl)
    <root>/manifest.json      ← 학습 완료 모델 추적

포맷은 모델별로 다르다("pth"는 개념적 의미이며 실제는 AutoGluon 디렉터리 또는 pkl).
zero-shot 모델(Chronos2/TimesFM)과 무상태 모델(LI/SeasonalNaive)은 저장이 필요 없어
manifest에 "stateless"로만 기록한다.
"""

from __future__ import annotations

import json
from pathlib import Path


class CheckpointManager:
    """모델 체크포인트의 경로·존재여부·manifest를 관리한다."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.manifest: dict = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

    def model_dir(self, model_name: str) -> Path:
        """모델별 저장 디렉터리(없으면 생성)."""
        d = self.root / model_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── AutoGluon (predictor 디렉터리) ──
    def autogluon_exists(self, model_name: str) -> bool:
        return (self.root / model_name / "predictor").exists()

    # ── CAFI (pkl) ──
    def cafi_exists(self, model_name: str) -> bool:
        return (self.root / model_name / "cafi_state.pkl").exists()

    def exists(self, model_name: str, kind: str) -> bool:
        """kind: 'autogluon' | 'cafi' | 'stateless'."""
        if kind == "autogluon":
            return self.autogluon_exists(model_name)
        if kind == "cafi":
            return self.cafi_exists(model_name)
        return True  # stateless는 항상 '준비됨'

    def record(self, model_name: str, kind: str, info: dict | None = None) -> None:
        """학습 완료를 manifest에 기록한다."""
        entry = {"kind": kind, "dir": str(self.root / model_name)}
        if info:
            entry.update(info)
        self.manifest[model_name] = entry
        self._save_manifest()

    def is_recorded(self, model_name: str) -> bool:
        return model_name in self.manifest

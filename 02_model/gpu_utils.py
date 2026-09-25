"""
gpu_utils.py
GPU 호환성 자동 선택

torch를 import하기 전에 반드시 이 모듈을 import해야 한다.
내부적으로 torch를 import하지 않고 nvidia-smi로 GPU를 감지하므로
CUDA_VISIBLE_DEVICES 설정이 torch 초기화 이전에 적용된다.

사용법:
    import gpu_utils        # run_experiment.py 맨 위에서 import
    import torch            # 이후 torch import
"""

from __future__ import annotations

import os
import subprocess


# ── PyTorch 빌드별 지원 SM 목록 (major, minor) ───────────────────────────────
# cu124 이하: SM 최대 9.0(Hopper)
# cu128 / nightly: SM 10.x(Blackwell GB200), 12.x(Blackwell B200) 포함
_SM_SUPPORTED_CU124 = {(5,0),(6,0),(6,1),(7,0),(7,5),(8,0),(8,6),(8,9),(9,0)}
_SM_SUPPORTED_CU128 = _SM_SUPPORTED_CU124 | {(10,0),(12,0)}


def _query_gpus() -> list[tuple[int, str, int, int]]:
    """
    nvidia-smi로 GPU 정보를 조회한다.

    Returns
    -------
    list of (index, name, sm_major, sm_minor)
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            idx = int(parts[0])
            name = parts[1]
            major, minor = parts[2].split(".")
            gpus.append((idx, name, int(major), int(minor)))
        return gpus
    except Exception:
        return []


def _detect_torch_cuda_version() -> str:
    """
    설치된 torch wheel의 CUDA 버전을 파일 이름 패턴으로 감지한다.
    torch 자체를 import하지 않고 site-packages 파일명을 확인한다.
    """
    import glob, sys
    for sp in sys.path:
        hits = glob.glob(os.path.join(sp, "torch-*.dist-info", "WHEEL"))
        if hits:
            try:
                with open(hits[0]) as f:
                    content = f.read()
                # Tag: cp312-cp312-linux_x86_64  (CPU) 또는
                # torch-2.6.0+cu124 → METADATA 확인
                pass
            except Exception:
                pass
        # METADATA 파일에서 버전 추출
        meta_hits = glob.glob(os.path.join(sp, "torch-*.dist-info", "METADATA"))
        for meta_path in meta_hits:
            try:
                with open(meta_path) as f:
                    for line in f:
                        if line.startswith("Version:"):
                            ver = line.split(":", 1)[1].strip()
                            if "cu128" in ver:
                                return "cu128"
                            if "cu126" in ver:
                                return "cu126"
                            if "cu124" in ver:
                                return "cu124"
                            if "cu121" in ver:
                                return "cu121"
                            if "cu118" in ver:
                                return "cu118"
                            return "cpu"
            except Exception:
                continue
    return "unknown"


def select_compatible_gpu(verbose: bool = True) -> None:
    """
    nvidia-smi로 GPU SM 버전을 확인하고, 설치된 PyTorch 빌드가
    지원하지 않는 GPU(Blackwell SM≥10 등)를 CUDA_VISIBLE_DEVICES로 제외한다.

    torch import 전에 호출되므로 내부에서 torch를 import하지 않는다.
    """
    # 이미 CUDA_VISIBLE_DEVICES가 명시적으로 설정되어 있으면 건드리지 않는다.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        if verbose:
            print(f"[GPU] CUDA_VISIBLE_DEVICES 이미 설정됨: {os.environ['CUDA_VISIBLE_DEVICES']}")
        return

    gpus = _query_gpus()
    if not gpus:
        return  # nvidia-smi 실패 → 그냥 통과

    cuda_ver = _detect_torch_cuda_version()
    if "128" in cuda_ver or "126" in cuda_ver:
        supported_sm = _SM_SUPPORTED_CU128
    else:
        supported_sm = _SM_SUPPORTED_CU124

    compat = []
    incompat = []
    for idx, name, major, minor in gpus:
        # major 버전이 지원 범위에 있거나 더 낮으면 호환
        is_ok = any(
            major < s_major or (major == s_major and minor <= s_minor)
            for s_major, s_minor in supported_sm
        ) or any(major == s_major for s_major, _ in supported_sm)

        # 더 간단한 기준: SM 10.0 이상이고 cu124 이하이면 비호환
        if major >= 10 and "128" not in cuda_ver and "126" not in cuda_ver:
            is_ok = False

        if is_ok:
            compat.append((idx, name, major, minor))
        else:
            incompat.append((idx, name, major, minor))

    if not incompat:
        return  # 모든 GPU 호환 → 아무것도 하지 않음

    if not compat:
        if verbose:
            print("[GPU] 경고: 호환 GPU 없음. CPU 모드로 실행됩니다.")
            print(f"[GPU] 설치된 PyTorch: {cuda_ver}, 최대 SM={max(supported_sm)}")
        return

    visible = ",".join(str(i) for i, _, _, _ in compat)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible

    if verbose:
        incompat_str = ", ".join(f"GPU{i} {n} (SM{m}.{mi})" for i, n, m, mi in incompat)
        compat_str  = ", ".join(f"GPU{i} {n} (SM{m}.{mi})" for i, n, m, mi in compat)
        print(f"[GPU] PyTorch 빌드: {cuda_ver}")
        print(f"[GPU] 비호환 제외: {incompat_str}")
        print(f"[GPU] 사용 GPU:    {compat_str}")
        print(f"[GPU] CUDA_VISIBLE_DEVICES={visible}")


# 모듈 import 시 자동 실행
select_compatible_gpu()

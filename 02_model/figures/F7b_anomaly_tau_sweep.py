"""
F7b_anomaly_tau_sweep.py
Figure 7b — 이상치 탐지 임계값(τ) 스윕: Precision-Recall 곡선.

F7에서 τ=3.5 고정 시 precision이 낮았던 이유(민감하지만 특이도 낮음)를 정량화한다.
잔차 스캔(scan_residuals)은 τ와 무관하므로 **모델당 1회만 스캔**하고, 같은 z-score에
여러 τ를 적용해 Precision/Recall/F1을 싸게 계산한다.

  (a) Precision–Recall 곡선 (τ를 1→8로 sweep, 모델별). τ=3.5 동작점 별표.
  (b) Precision/Recall/F1 vs τ (모델별; 최적 F1 지점 표시)

평가 단위: 모든 타깃 변수의 (시점) 샘플을 풀링해 모델당 하나의 곡선.

입력 : 01_data/*.xlsx, split.json, results.csv, 03_result/comparison/models/
출력 : 03_result/comparison/figures/F7b_anomaly_tau_sweep.{pdf,png}

실행:
  python F7b_anomaly_tau_sweep.py
  python F7b_anomaly_tau_sweep.py --models CAFI,TimesFM2.5,Chronos2 --n-inject 20
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent
for _p in (_HERE, _MODEL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gpu_utils  # noqa: F401
import fig_utils as fu
from preprocessing import preprocess_file, TARGET_COLS
from anomaly_detection import scan_residuals, mad_zscore, inject_synthetic_spikes

from F7_anomaly_detection import _build_models, DEFAULT_MODELS
from F6_imputation_examples import _pick_representative

TARGET_VARS = list(TARGET_COLS.values())
TAU_OPERATING = 3.5   # F7에서 쓴 기본 동작점


def _pr_at_tau(z_all: np.ndarray, truth_all: np.ndarray, taus: np.ndarray):
    """풀링된 (z, truth)에서 τ별 precision/recall/f1 배열."""
    prec, rec, f1 = [], [], []
    finite = np.isfinite(z_all)
    for t in taus:
        pred = finite & (z_all > t)
        tp = int(np.sum(pred & truth_all))
        fp = int(np.sum(pred & ~truth_all))
        fn = int(np.sum(~pred & truth_all))
        p = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        r = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        prec.append(p); rec.append(r)
        f1.append(2 * p * r / (p + r) if (p and r and (p + r) > 0) else np.nan)
    return np.array(prec), np.array(rec), np.array(f1)


def generate(result_dir=None, greenhouse=None, model_keys=None,
             n_inject=20, block_h=6, context_len=720, seed=42,
             tau_min=1.0, tau_max=8.0, tau_steps=40) -> None:
    rd = Path(result_dir) if result_dir else fu.get_comparison_dir()
    split = json.load(open(rd / "split.json"))
    test_paths = [Path(p) for p in split["test"]]
    test_stems = [p.stem for p in test_paths]
    model_keys = model_keys or DEFAULT_MODELS

    if greenhouse is None:
        greenhouse = _pick_representative(rd, test_stems)
    if greenhouse not in test_stems:
        warnings.warn(f"{greenhouse} test 셋에 없음 → 첫 test 온실 사용")
        greenhouse = test_stems[0]
    res = preprocess_file(test_paths[test_stems.index(greenhouse)], include_covariates=True)
    data = res["data"]; name = res["name"]
    tcols = [c for c in TARGET_VARS if c in data.columns]
    ccols = [c for c in data.columns if c not in TARGET_VARS]
    valid = data[tcols].notna()

    # 전 타깃에 합성 스파이크 주입(모델 공통)
    injected_frame = data.copy(); injected_pos = {}
    for v in tcols:
        s_inj, pos = inject_synthetic_spikes(data[v], valid[v], n_inject=n_inject, seed=seed)
        injected_frame[v] = s_inj; injected_pos[v] = pos

    models = _build_models(model_keys, context_len, rd / "models")
    taus = np.linspace(tau_min, tau_max, tau_steps)

    curves = {}  # model -> (prec, rec, f1)
    for k, model in models.items():
        frame = injected_frame[tcols + ccols] if (k == "CAFI" and ccols) else injected_frame[tcols]
        try:
            residual = scan_residuals(model, frame, tcols, block_h=block_h)  # 모델당 1회
        except Exception as e:
            warnings.warn(f"[{k}] scan 실패 → 생략: {e}")
            continue
        z_pool, t_pool = [], []
        for v in tcols:
            z = mad_zscore(residual[v]).values.astype(float)
            truth = np.zeros(len(data), dtype=bool)
            truth[injected_pos[v]["start"].astype(int).values] = True
            z_pool.append(z); t_pool.append(truth)
        z_all = np.concatenate(z_pool); truth_all = np.concatenate(t_pool)
        curves[k] = _pr_at_tau(z_all, truth_all, taus)

    mdl_list = fu.ordered_models(curves.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) PR 곡선
    ax = axes[0]
    op_idx = int(np.argmin(np.abs(taus - TAU_OPERATING)))
    for k in mdl_list:
        prec, rec, f1 = curves[k]
        ax.plot(rec, prec, color=fu.cmp_model_color(k), lw=fu.cmp_lw(k),
                marker=fu.cmp_model_marker(k), markersize=3, label=fu.cmp_model_display(k),
                zorder=6 if k == "CAFI" else 4)
        # τ=3.5 동작점
        if np.isfinite(rec[op_idx]) and np.isfinite(prec[op_idx]):
            ax.scatter([rec[op_idx]], [prec[op_idx]], s=80, facecolors="none",
                       edgecolors=fu.cmp_model_color(k), linewidths=1.6, zorder=8)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_title(f"(a) Precision–Recall (τ swept {taus[0]:g}→{taus[-1]:g})\n"
                 f"○ = operating point τ={TAU_OPERATING}")
    ax.grid(True, alpha=0.3, lw=0.5); ax.legend(fontsize=8, loc="upper right")

    # (b) F1 vs τ
    ax = axes[1]
    for k in mdl_list:
        prec, rec, f1 = curves[k]
        ax.plot(taus, f1, color=fu.cmp_model_color(k), lw=fu.cmp_lw(k),
                label=fu.cmp_model_display(k), zorder=6 if k == "CAFI" else 4)
        if np.isfinite(f1).any():
            bi = int(np.nanargmax(f1))
            ax.scatter([taus[bi]], [f1[bi]], s=60, color=fu.cmp_model_color(k), zorder=8)
            ax.annotate(f"τ*={taus[bi]:.1f}", (taus[bi], f1[bi]), fontsize=7,
                        textcoords="offset points", xytext=(3, 4))
    ax.axvline(TAU_OPERATING, color="#888", ls=":", lw=1.0)
    ax.text(TAU_OPERATING, ax.get_ylim()[1] * 0.95, " τ=3.5", fontsize=7, color="#888")
    ax.set_xlabel("τ (MAD z-score threshold)"); ax.set_ylabel("F1")
    ax.set_title("(b) F1 vs τ  (★ = best F1)")
    ax.grid(True, alpha=0.3, lw=0.5); ax.legend(fontsize=8)

    fig.suptitle(f"Anomaly detection τ-sweep — {name} (synthetic spikes, n={n_inject}/var, "
                 f"pooled over {len(tcols)} variables)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fu.save_figure(fig, rd / "figures", "F7b_anomaly_tau_sweep")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="F7b 이상치 탐지 τ-sweep PR 곡선")
    p.add_argument("--result-dir", type=str, default=str(fu.get_comparison_dir()))
    p.add_argument("--greenhouse", type=str, default=None)
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument("--n-inject", type=int, default=20)
    p.add_argument("--tau-min", type=float, default=1.0)
    p.add_argument("--tau-max", type=float, default=8.0)
    p.add_argument("--tau-steps", type=int, default=40)
    args = p.parse_args()
    mk = [m.strip() for m in args.models.split(",") if m.strip()]
    generate(Path(args.result_dir), greenhouse=args.greenhouse, model_keys=mk,
             n_inject=args.n_inject, tau_min=args.tau_min, tau_max=args.tau_max,
             tau_steps=args.tau_steps)

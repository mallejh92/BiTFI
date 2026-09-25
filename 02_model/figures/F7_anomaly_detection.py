"""
F7_anomaly_detection.py
Figure 7 — 이상치(스파이크) 탐지 및 복원 성능.

합성 스파이크를 알려진 위치에 주입한 뒤, 각 모델의 잔차 기반 탐지(Precision/
Recall/F1)와 복원 정확도(주입 전 참값 대비 MAE)를 평가한다.

  (a) 모델별 탐지 Precision / Recall / F1 (5개 타깃 변수 합산/평균)
  (b) 모델별 복원 MAE (정규화 스케일)
  (c) 예시: 한 변수 시계열에 주입된 스파이크(✕)와 탐지된 위치(○)

학습형 모델은 03_result/comparison/models/ 체크포인트에서 로드, CAFI는 v1 zero-shot.

입력 : 01_data/*.xlsx, split.json, 03_result/comparison/models/
출력 : 03_result/comparison/figures/F7_anomaly_detection.{pdf,png}

실행:
  python F7_anomaly_detection.py
  python F7_anomaly_detection.py --greenhouse PF_0020209_01 --models CAFI,TimesFM2.5,Chronos2
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
from anomaly_detection import (
    scan_residuals, mad_zscore, spike_flag,
    inject_synthetic_spikes, evaluate_detection,
    restore_flagged, restoration_error,
)

TARGET_VARS = list(TARGET_COLS.values())
AG_SPECS = {"AG-DeepAR": {"DeepAR": {}}, "AG-LightGBM": {"RecursiveTabular": {"model_name": "GBM"}}}
DEFAULT_MODELS = ["LI", "SeasonalNaive", "AG-LightGBM", "AG-RandomForest",
                  "AG-DeepAR", "AG-PatchTST", "Chronos2", "TimesFM2.5", "CAFI"]


def _build_models(model_keys, context_len, models_dir):
    from models.foundation_model import ChronosImputation, TimesFMImputation, CAFIImputation
    from models.linear_interpolation import LinearInterpolation
    from models.seasonal_naive import SeasonalNaiveImputation
    models = {}
    for k in model_keys:
        try:
            if k == "LI":
                models[k] = LinearInterpolation()
            elif k == "SeasonalNaive":
                models[k] = SeasonalNaiveImputation(season_length=24, context_len=context_len)
            elif k == "CAFI":
                models[k] = CAFIImputation(fine_tune=False, context_len=context_len, name="CAFI")
            elif k == "Chronos2":
                models[k] = ChronosImputation(context_len=context_len, fine_tune=False)
            elif k == "TimesFM2.5":
                models[k] = TimesFMImputation(context_len=context_len)
            elif k in AG_SPECS:
                from models.autogluon_model import AutoGluonImputation
                mdir = models_dir / k
                if mdir.exists():
                    models[k] = AutoGluonImputation.load(
                        mdir, context_len=context_len, hyperparameters=AG_SPECS[k], name=k)
                else:
                    warnings.warn(f"[{k}] 체크포인트 없음 → 생략")
        except Exception as e:
            warnings.warn(f"[{k}] 생성 실패 → 생략: {e}")
    return models


def generate(result_dir=None, greenhouse=None, model_keys=None,
             n_inject=20, tau=3.5, block_h=6, context_len=720, seed=42) -> None:
    rd = Path(result_dir) if result_dir else fu.get_comparison_dir()
    split = json.load(open(rd / "split.json"))
    model_keys = model_keys or DEFAULT_MODELS

    target_path = None
    for p in split["test"]:
        if greenhouse is None or Path(p).stem == greenhouse:
            target_path = Path(p)
            if greenhouse is not None:
                break
    res = preprocess_file(target_path, include_covariates=True)
    data = res["data"]; name = res["name"]
    tcols = [c for c in TARGET_VARS if c in data.columns]
    ccols = [c for c in data.columns if c not in TARGET_VARS]
    valid = data[tcols].notna()

    models = _build_models(model_keys, context_len, rd / "models")

    # 합성 스파이크를 모든 타깃 변수에 한 번에 주입(모델 간 동일 위치 사용).
    # 스파이크는 희소(변수당 ~n_inject개)하므로 CAFI covariate 교차오염은 무시 가능하며,
    # 이렇게 하면 scan_residuals/restore_flagged를 모델당 1회로 줄여 비용을 크게 낮춘다.
    injected_frame = data.copy()
    injected_pos = {}
    for v in tcols:
        s_inj, pos = inject_synthetic_spikes(data[v], valid[v], n_inject=n_inject, seed=seed)
        injected_frame[v] = s_inj
        injected_pos[v] = pos

    det = {k: {"precision": [], "recall": [], "f1": []} for k in models}
    rest = {k: [] for k in models}
    roc_data = {}   # model -> (z_all, truth_all) 풀링 (ROC용)
    example = None  # (var, series, injected_pos, flag)

    for k, model in models.items():
        frame = (injected_frame[tcols + ccols] if (k == "CAFI" and ccols)
                 else injected_frame[tcols]).copy()
        try:
            residual = scan_residuals(model, frame, tcols, block_h=block_h)  # 모델당 1회
        except Exception as e:
            warnings.warn(f"[{k}] scan 실패 → 생략: {e}")
            continue
        flagged = pd.DataFrame(False, index=data.index, columns=tcols)
        z_pool, t_pool = [], []
        for v in tcols:
            zser = mad_zscore(residual[v]); flag = spike_flag(zser, tau=tau)
            flagged[v] = flag.values
            m = evaluate_detection(flag, injected_pos[v], n=len(data))
            for key in ("precision", "recall", "f1"):
                if not np.isnan(m[key]):
                    det[k][key].append(m[key])
            truth = np.zeros(len(data), dtype=bool)
            truth[injected_pos[v]["start"].astype(int).values] = True
            z_pool.append(zser.values.astype(float)); t_pool.append(truth)
            if example is None and k == "CAFI":
                example = (v, injected_frame[v], injected_pos[v], flag)
        roc_data[k] = (np.concatenate(z_pool), np.concatenate(t_pool))
        try:
            restored = restore_flagged(model, frame, tcols, flagged)  # 모델당 1회
            for v in tcols:
                std = float(data[v].std()) or 1.0
                rerr = restoration_error(restored[v], injected_pos[v])
                if not np.isnan(rerr["mae"]):
                    rest[k].append(rerr["mae"] / std)   # 표준편차로 정규화
        except Exception as e:
            warnings.warn(f"[{k}] restore 실패: {e}")

    mdl_list = fu.ordered_models(roc_data.keys())

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.26)

    # (a) ROC 곡선 (τ sweep, 모델별 AUC)
    ax = fig.add_subplot(gs[0, 0])
    roc_taus = np.concatenate([[0.0], np.linspace(0.5, 30.0, 60), [1e9]])
    for k in mdl_list:
        z_all, truth_all = roc_data[k]
        finite = np.isfinite(z_all)
        P = int(truth_all.sum()); N = int((~truth_all).sum())
        tpr, fpr = [], []
        for t in roc_taus:
            pred = finite & (z_all > t)
            tpr.append((pred & truth_all).sum() / P if P else np.nan)
            fpr.append((pred & ~truth_all).sum() / N if N else np.nan)
        fpr = np.array(fpr); tpr = np.array(tpr)
        order = np.argsort(fpr)
        auc = float(np.trapz(tpr[order], fpr[order]))
        ax.plot(fpr, tpr, color=fu.cmp_model_color(k), lw=fu.cmp_lw(k),
                label=f"{fu.cmp_model_display(k)} (AUC={auc:.2f})",
                zorder=6 if k == "CAFI" else 4)
    ax.plot([0, 1], [0, 1], color="#999", ls="--", lw=0.8, label="chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.set_title("(a) ROC (τ swept)"); ax.legend(fontsize=7, loc="lower right")

    # (b) P/R/F1 (운영 τ)
    ax = fig.add_subplot(gs[0, 1])
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(metrics)); width = 0.8 / max(len(mdl_list), 1)
    for i, k in enumerate(mdl_list):
        vals = [np.mean(det[k][mm]) if det[k][mm] else np.nan for mm in metrics]
        ax.bar(x + i * width, vals, width, label=fu.cmp_model_display(k),
               color=fu.cmp_model_color(k), edgecolor="black", linewidth=0.4)
    ax.set_xticks(x + width * (len(mdl_list) - 1) / 2)
    ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title(f"(b) Spike detection (τ={tau})"); ax.legend(fontsize=6, ncol=2)

    # (c) restoration MAE
    ax = fig.add_subplot(gs[1, 0])
    ys = [np.mean(rest[k]) if rest[k] else np.nan for k in mdl_list]
    ax.bar(range(len(mdl_list)), ys, color=[fu.cmp_model_color(k) for k in mdl_list],
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(mdl_list)))
    ax.set_xticklabels([fu.cmp_model_display(k) for k in mdl_list], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("restoration MAE / σ"); ax.set_title("(c) Restoration error")

    # (d) example timeline
    ax = fig.add_subplot(gs[1, 1])
    if example is not None:
        v, s_inj, pos, flag = example
        win = slice(0, min(len(s_inj), 1000))
        idx = np.arange(len(s_inj))[win]
        ax.plot(idx, s_inj.values[win], color="#444", lw=0.6, label="series (with spikes)")
        ip = pos["start"].astype(int).values
        ip = ip[ip < (win.stop or len(s_inj))]
        ax.scatter(ip, s_inj.values[ip], marker="x", color="red", s=40, label="injected", zorder=5)
        fl = np.where(flag.values[win])[0]
        ax.scatter(fl, s_inj.values[fl], marker="o", facecolors="none", edgecolors="blue",
                   s=60, label="detected", zorder=4)
        v_label = fu.VAR_SHORT_LABELS.get(v, v)
        ax.set_title(f"(d) Example (CAFI, {v_label}, τ={tau})"); ax.legend(fontsize=7)
        ax.set_xlabel("time index")
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "예시 없음", ha="center")

    fig.suptitle(f"Anomaly detection & restoration — {name} (synthetic spikes, n={n_inject}/var, "
                 f"all models)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fu.save_figure(fig, rd / "figures", "F7_anomaly_detection")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="F7 이상치 탐지/복원")
    p.add_argument("--result-dir", type=str, default=str(fu.get_comparison_dir()))
    p.add_argument("--greenhouse", type=str, default=None)
    p.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    p.add_argument("--n-inject", type=int, default=20)
    p.add_argument("--tau", type=float, default=3.5)
    args = p.parse_args()
    mk = [m.strip() for m in args.models.split(",") if m.strip()]
    generate(Path(args.result_dir), greenhouse=args.greenhouse, model_keys=mk,
             n_inject=args.n_inject, tau=args.tau)

"""
visualize.py
실험 결과 시각화 모듈 (논문 Figure 4, 5, 7, 8 스타일)

Figure 1: 모델별 × 변수별 scatter plot (측정값 vs 보간값)
Figure 2: loss_rate × R²/RMSE 선 그래프 (변수별 subplot)
Figure 3: 시계열 비교 (실측 vs 보간)
Figure 4: Ablation (zero-shot vs fine-tuned, 온실별 boxplot)

저장: results/figures/ 에 PNG + PDF
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경 대비

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── Publication-quality global style ──────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "mathtext.default": "regular",
})

VARIABLES = ["Tin", "Tout", "RH", "CO2", "Rad"]
VAR_LABELS = {
    "Tin": "Internal Temp (°C)",
    "Tout": "External Temp (°C)",
    "RH": "Humidity (%)",
    "CO2": r"CO$_2$ (ppm)",
    "Rad": "Solar Radiation (W/m²)",
}

# Seaborn deep palette — 논문 인쇄 시 구분 용이한 8색 배정
MODEL_COLORS = {
    "LI":              "#4C72B0",   # deep blue
    "LSTM":            "#DD8452",   # deep orange
    "PatchTST":        "#55A868",   # deep green
    "TimesFM2.5":      "#C44E52",   # deep red
    "TimesFM":         "#C44E52",
    "Chronos2":        "#937860",   # deep brown
    "CAFI":            "#8C8C8C",   # deep gray
    "CAFI2-zero":      "#17BECF",   # teal
    "Chronos-zero":    "#8172B3",
    "Chronos-ft":      "#937860",
}

# 논문 subplot 고정 순서
MODEL_ORDER = [
    "LI", "LSTM", "PatchTST", "TimesFM2.5",
    "Chronos2", "CAFI",
]

FIGURE_DIR = Path("results/figures")
DPI = 300
# 결과/시각화에서 제외할 zero-shot/보조 모델명을 한곳에서 관리한다.
EXCLUDED_MODEL_NAMES = {"cafi-zero", "cafi2-zero", "chronos2-zero"}
MODEL_RENAME = {"Chronos2-ft": "Chronos2", "CAFI-ft": "CAFI"}


def _is_excluded_model(model_name: str) -> bool:
    # 모델명이 제외 목록에 있으면 그래프에서 숨긴다.
    return str(model_name).strip().lower() in EXCLUDED_MODEL_NAMES


def _filter_models_df(df: pd.DataFrame) -> pd.DataFrame:
    # 집계 DataFrame에서 제외 모델을 제거해 후속 plot 함수가 공통 규칙을 따른다.
    if df.empty or "model" not in df.columns:
        return df
    filtered = df[~df["model"].astype(str).map(_is_excluded_model)].copy()
    filtered["model"] = filtered["model"].replace(MODEL_RENAME)
    return filtered


def _get_color(model_name: str) -> str:
    for k, c in MODEL_COLORS.items():
        if k.lower() in model_name.lower():
            return c
    return "#333333"


def _fig1_model_order_from_metrics(
    metrics_df: pd.DataFrame,
    mask_type: str,
    loss_rate: float,
) -> list[str]:
    """
    fig1과 동일한 mask/loss_rate 구간에서 results_all.csv에 등장하는 model 열 순서(첫 등장 순)를 반환한다.
    """
    sub = metrics_df[
        (metrics_df["mask_type"] == mask_type)
        & (np.isclose(metrics_df["loss_rate"].astype(float), float(loss_rate), rtol=0, atol=1e-9))
    ]
    if sub.empty:
        return []
    # CSV에서의 등장 순서를 유지한 채 모델명만 유니크하게 수집한다.
    return list(dict.fromkeys(sub["model"].astype(str).tolist()))


def save_fig(fig: plt.Figure, filename: str, fig_dir: Path = FIGURE_DIR) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = fig_dir / f"{filename}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {fig_dir / filename}.{{png,pdf}}")


# ──────────────────────────────────────────────
# Figure 1: Scatter plot (논문 Figure 4)
# ──────────────────────────────────────────────

def plot_scatter(
    pred_dict: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    fig_dir: Path = FIGURE_DIR,
    mask_type: str = "individual",
    loss_rate: float = 0.30,
    model_order: list[str] | None = None,
) -> None:
    """
    pred_dict: {model_name: {var_name: (y_true, y_pred)}}  — 역정규화된 실제 단위값
    model_order: metrics_df(results_all.csv)의 model 열 순서와 동일하게 열을 배치할 때 사용한다.
    """
    # 제외 대상 모델은 산점도에서 제거한다.
    pred_dict = {k: v for k, v in pred_dict.items() if not _is_excluded_model(k)}
    if not pred_dict:
        print("  [Fig1] no eligible models after exclusion — skipped")
        return
    if model_order is not None:
        models = [m for m in model_order if m in pred_dict]
    else:
        models = [m for m in MODEL_ORDER if m in pred_dict]
    if not models:
        # 순서 템플릿과 무관하게 남은 모델이 있으면 데이터 등장 순으로 표시한다.
        models = list(pred_dict.keys())
    vars_  = VARIABLES

    fig, axes = plt.subplots(
        len(vars_), len(models),
        figsize=(3.5 * len(models), 3.5 * len(vars_)),
        squeeze=False,
    )

    for j, model in enumerate(models):
        per_model = pred_dict.get(model, {})
        for i, var in enumerate(vars_):
            ax = axes[i][j]
            if var not in per_model:
                ax.axis("off")
                continue
            y_true, y_pred = per_model[var]
            color = _get_color(model)

            ax.scatter(y_true, y_pred, s=4, alpha=0.4, color=color, rasterized=True)

            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
            margin = (hi - lo) * 0.05 if hi > lo else 1.0
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)

            from sklearn.metrics import r2_score
            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
            r2 = r2_score(y_true[valid], y_pred[valid]) if valid.sum() > 1 else np.nan
            ax.text(
                0.05, 0.92, f"$R^2$={r2:.3f}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment="top",
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1.0},
            )
            if j == 0:
                # y축: 실제 단위 포함 변수 레이블
                ax.set_ylabel(VAR_LABELS.get(var, var), fontsize=8)
            if i == 0:
                ax.set_title(model, fontsize=9, pad=4)
            # x/y축 모두 실제 단위
            unit_label = VAR_LABELS.get(var, var)
            ax.set_xlabel(f"Measured ({unit_label})", fontsize=7)
            ax.set_ylabel(f"Imputed ({unit_label})", fontsize=7)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    save_fig(fig, f"fig1_scatter_{mask_type}_lr{int(loss_rate*100):02d}", fig_dir)


# ──────────────────────────────────────────────
# Figure 2: loss_rate × performance (논문 Figure 5)
# ──────────────────────────────────────────────

def plot_loss_rate_curves(
    metrics_df: pd.DataFrame,
    fig_dir: Path = FIGURE_DIR,
    mask_type: str = "individual",
) -> None:
    """
    온실별 MSE/MAE를 loss_rate별로 평균(±1σ 신뢰대역)으로 표시.

    metrics_df에는 온실 × loss_rate × model 조합이 여러 행으로 존재한다.
    단순 plot이면 같은 loss_rate 내 여러 온실 점이 지그재그로 연결되므로
    반드시 groupby(model, loss_rate).mean()으로 집계 후 플롯해야 한다.
    """
    df = metrics_df[metrics_df["mask_type"] == mask_type].copy()
    df = _filter_models_df(df)
    if df.empty:
        print(f"  [Fig2] {mask_type}: no eligible models after exclusion — skipped")
        return
    models_in_data = set(df["model"].unique())
    models = [m for m in MODEL_ORDER if m in models_in_data]
    if not models:
        models = sorted(models_in_data)
    vars_  = [v for v in VARIABLES if v in df["variable"].unique()]

    # 존재하는 metric 컬럼만 사용
    available_metrics = [m for m in ["MSE", "MAE"] if m in df.columns]
    if not available_metrics:
        print("  [Fig2] MSE/MAE 컬럼 없음 — 건너뜀")
        return

    fig, axes = plt.subplots(
        len(available_metrics), len(vars_),
        figsize=(3.5 * len(vars_), 4 * len(available_metrics)),
        squeeze=False,
    )

    for j, var in enumerate(vars_):
        sub = df[df["variable"] == var]
        for metric_idx, metric in enumerate(available_metrics):
            ax = axes[metric_idx][j]
            for model in models:
                m_sub = sub[sub["model"] == model]
                if m_sub.empty or metric not in m_sub.columns:
                    continue

                # 온실별 평균 + 표준편차 (loss_rate 단위 집계)
                agg = (
                    m_sub.groupby("loss_rate")[metric]
                    .agg(mean="mean", std="std")
                    .reset_index()
                    .sort_values("loss_rate")
                )
                xs = agg["loss_rate"] * 100
                ys = agg["mean"]
                errs = agg["std"].fillna(0)

                color = _get_color(model)
                ax.plot(xs, ys, marker="o", markersize=4, label=model, color=color)
                ax.fill_between(xs, ys - errs, ys + errs, alpha=0.12, color=color)

            if j == 0:
                ax.set_ylabel(metric, fontsize=9)
            if metric_idx == len(available_metrics) - 1:
                ax.set_xlabel("Loss Rate (%)", fontsize=8)
            if metric_idx == 0:
                ax.set_title(VAR_LABELS.get(var, var), fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            # legend는 첫 번째 행의 마지막 컬럼에만 표시
            if metric_idx == 0 and j == len(vars_) - 1:
                ax.legend(fontsize=7, loc="best", frameon=False)

    plt.tight_layout()
    save_fig(fig, f"fig2_loss_rate_curves_{mask_type}", fig_dir)


# ──────────────────────────────────────────────
# Figure 3: 시계열 비교 (논문 Figure 7, 8)
# ──────────────────────────────────────────────

def plot_timeseries(
    ground_truth: pd.DataFrame,
    imputed_dict: dict[str, pd.DataFrame],
    mask_matrix: pd.DataFrame,
    variable: str = "Tin",
    n_points: int = 336,    # 2주 (시간 단위)
    fig_dir: Path = FIGURE_DIR,
    suffix: str = "",
    scalers: dict | None = None,
    meta: dict | None = None,
) -> None:
    """
    실측값(회색) vs 각 모델 보간값(색상) 시계열 비교.
    scalers: 역정규화용 scaler dict {col: MinMaxScaler}
    meta: {'gh_name', 'freq', 'loss_rate', 'test_ratio', 'n_total'} — Fig 3 설명 주석용
    """
    if variable not in ground_truth.columns:
        return
    # 제외 대상 모델은 시계열 비교에서 제거한다.
    imputed_dict = {k: v for k, v in imputed_dict.items() if not _is_excluded_model(k)}
    if not imputed_dict:
        print(f"  [Fig3] {variable}: no eligible models after exclusion — skipped")
        return

    # ── 역정규화 ─────────────────────────────────
    def _inv(series: pd.Series) -> pd.Series:
        if scalers is None:
            return series
        scaler = scalers.get(variable)
        if scaler is None:
            return series
        vals = series.values.reshape(-1, 1)
        return pd.Series(
            scaler.inverse_transform(vals).flatten(),
            index=series.index,
        )

    gt_series   = _inv(ground_truth[variable].iloc[:n_points])
    mask_series = mask_matrix[variable].iloc[:n_points] == 0  # True=결측

    models_in_data = set(imputed_dict.keys())
    models = [m for m in MODEL_ORDER if m in models_in_data]
    if not models:
        models = list(imputed_dict.keys())
    fig, axes = plt.subplots(
        len(models), 1,
        figsize=(14, 3 * len(models)),
        sharex=True, squeeze=False,
    )

    x = np.arange(len(gt_series))

    for idx, model in enumerate(models):
        ax = axes[idx][0]
        imp_series = _inv(imputed_dict[model][variable].iloc[:n_points])

        ax.plot(x, gt_series.values, color="#AAAAAA", lw=1.0, label="Measured")

        imp_masked = imp_series.where(mask_series).values
        ax.plot(x, imp_masked, color=_get_color(model), lw=1.2, label=f"{model} (imputed)")

        ylim = ax.get_ylim()
        ax.fill_between(
            x, ylim[0], ylim[1],
            where=mask_series.values,
            alpha=0.08, color="red", label="Missing"
        )

        ax.set_ylabel(f"{model}\n{VAR_LABELS.get(variable, variable)}", fontsize=8)
        ax.tick_params(labelsize=7)
        # legend는 첫 번째 subplot에만 표시
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right", frameon=False)
        ax.grid(True, alpha=0.2)

    axes[-1][0].set_xlabel("Time Steps (hours)", fontsize=9)

    plt.tight_layout()
    save_fig(fig, f"fig3_timeseries_{variable}{('_' + suffix) if suffix else ''}", fig_dir)


# ──────────────────────────────────────────────
# Figure 4: Ablation — zero-shot vs fine-tuned
# ──────────────────────────────────────────────

def plot_ablation_sensitivity(
    metrics_df: pd.DataFrame,
    fig_dir: Path = FIGURE_DIR,
) -> None:
    """
    Ablation: 결측률(loss_rate) × 마스킹 유형에 따른 모델별 성능 민감도 분석.

    설계 의도:
      - "결측이 심해질수록 각 모델이 얼마나 빨리 나빠지는가?"를 정량화하는 ablation
      - NMAE(Normalized MAE)를 사용하여 변수 간 스케일 차이를 제거하고
        모든 변수를 하나의 축으로 통합 비교
      - loss_rate별 grouped bar chart + 1σ error bar (온실 간 분산)
      - mask_type별 subplot (block / individual)

    Ablation 해석:
      - 낮은 loss_rate에서 모든 모델이 비슷하면 → 쉬운 조건에서 차별화 없음
      - 높은 loss_rate에서 격차가 벌어질수록 → 해당 모델의 강건성 열위
      - 모델 A가 B보다 loss_rate 상승에 둔감하면 → A가 더 robust
    """
    metrics_df = _filter_models_df(metrics_df)
    if metrics_df.empty:
        print("  [Fig4] no eligible models after exclusion — skipped")
        return
    # NMAE 우선, 없으면 MAE, 없으면 MSE
    metric = next((m for m in ["NMAE", "MAE", "MSE"] if m in metrics_df.columns), None)
    if metric is None:
        print("  [Fig4] 평가 지표 컬럼 없음 — 건너뜀")
        return

    models_all = set(metrics_df["model"].unique())
    models = [m for m in MODEL_ORDER if m in models_all]
    if not models:
        models = sorted(models_all)

    mask_types = sorted(metrics_df["mask_type"].unique())
    loss_rates = sorted(metrics_df["loss_rate"].astype(float).unique())

    if not loss_rates or not models:
        return

    n_cols = len(mask_types)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), squeeze=False)

    bar_width = 0.8 / max(len(models), 1)
    x = np.arange(len(loss_rates))

    for col_idx, mask_type in enumerate(mask_types):
        ax = axes[0][col_idx]
        df_sub = metrics_df[metrics_df["mask_type"] == mask_type]

        for m_idx, model in enumerate(models):
            m_sub = df_sub[df_sub["model"] == model]
            means, errs = [], []
            for lr in loss_rates:
                lr_rows = m_sub[
                    np.isclose(m_sub["loss_rate"].astype(float), float(lr), rtol=0, atol=1e-9)
                ]
                val = lr_rows[metric].dropna()
                means.append(float(val.mean()) if len(val) > 0 else np.nan)
                errs.append(float(val.std()) if len(val) > 1 else 0.0)

            offset = (m_idx - len(models) / 2 + 0.5) * bar_width
            ax.bar(
                x + offset, means, bar_width * 0.9,
                label=model,
                color=_get_color(model),
                alpha=0.85,
                yerr=errs,
                capsize=2,
                error_kw={"linewidth": 0.8, "ecolor": "#333333"},
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(lr * 100)}%" for lr in loss_rates], fontsize=8)
        ax.set_xlabel("Loss Rate", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        if col_idx == 0:
            ax.set_ylabel(f"{metric} (lower is better)", fontsize=9)
            ax.legend(fontsize=8, loc="upper left", frameon=False)
    plt.tight_layout()
    save_fig(fig, "fig4_ablation_sensitivity", fig_dir)


# ──────────────────────────────────────────────
# Subgroup Figures: season / metadata comparison
# ──────────────────────────────────────────────

SUBGROUP_FIGURE_CONFIG = {
    "season": {
        "filename": "fig_subgroup_season_nmae",
        "xlabel": "Season",
        "order": ["Spring", "Summer", "Autumn", "Winter"],
    },
    "structure_type": {
        "filename": "fig_subgroup_structure_nmae",
        "xlabel": "Structure Type",
        "order": None,
    },
    "Covering material": {
        "filename": "fig_subgroup_covering_nmae",
        "xlabel": "Covering Material",
        "order": None,
    },
    "area_group": {
        "filename": "fig_subgroup_area_nmae",
        "xlabel": "Area Group",
        "order": ["<1000 m2", "1000-4999 m2", ">9000 m2"],
    },
}


def _ordered_models_from_df(df: pd.DataFrame) -> list[str]:
    # 논문 Figure에서 모델 색상과 순서가 일관되도록 기존 MODEL_ORDER를 우선 적용한다.
    df = _filter_models_df(df)
    models_in_data = set(df["model"].astype(str).unique())
    models = [m for m in MODEL_ORDER if m in models_in_data]
    models.extend(sorted(models_in_data - set(models)))
    return models


def _ordered_group_values(df: pd.DataFrame, group_type: str) -> list[str]:
    # season과 area_group은 논문 해석 순서를 고정하고 나머지는 데이터에 등장한 순서를 사용한다.
    config = SUBGROUP_FIGURE_CONFIG[group_type]
    values_in_data = list(dict.fromkeys(df["group_value"].astype(str).tolist()))
    if config["order"] is None:
        return values_in_data
    ordered = [v for v in config["order"] if v in values_in_data]
    ordered.extend([v for v in values_in_data if v not in ordered])
    return ordered


def plot_subgroup_nmae(
    subgroup_metrics_df: pd.DataFrame,
    group_type: str,
    fig_dir: Path = FIGURE_DIR,
) -> None:
    """
    서브그룹별 NMAE를 모델별 grouped bar plot으로 비교한다.
    """
    if group_type not in SUBGROUP_FIGURE_CONFIG:
        return

    df = subgroup_metrics_df[
        (subgroup_metrics_df["group_type"] == group_type)
        & (subgroup_metrics_df["n_eval"] > 0)
    ].copy()
    df = _filter_models_df(df)
    df = df.dropna(subset=["NMAE", "group_value"])
    if df.empty:
        print(f"  [Subgroup] {group_type}: no valid NMAE rows — skipped")
        return

    # 온실·변수·마스킹 조건 전체를 평균해 서브그룹별 모델 성능을 요약한다.
    agg = (
        df.groupby(["group_value", "model"])["NMAE"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )
    agg["sem"] = agg["sem"].fillna(0.0)

    group_values = _ordered_group_values(df, group_type)
    models = _ordered_models_from_df(df)
    x = np.arange(len(group_values), dtype=float)
    width = min(0.80 / max(len(models), 1), 0.16)

    fig_width = max(7.0, 1.3 * len(group_values) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))

    for idx, model in enumerate(models):
        model_agg = agg[agg["model"] == model].set_index("group_value")
        means = [model_agg.loc[g, "mean"] if g in model_agg.index else np.nan for g in group_values]
        errors = [model_agg.loc[g, "sem"] if g in model_agg.index else 0.0 for g in group_values]
        offset = (idx - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=errors,
            capsize=2,
            color=_get_color(model),
            label=model,
            alpha=0.85,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )

    ax.set_xlabel(SUBGROUP_FIGURE_CONFIG[group_type]["xlabel"], fontsize=9)
    ax.set_ylabel("NMAE", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(group_values, rotation=20, ha="right")
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    plt.tight_layout()
    save_fig(fig, SUBGROUP_FIGURE_CONFIG[group_type]["filename"], fig_dir)


# ──────────────────────────────────────────────
# 전체 시각화 실행 헬퍼
# ──────────────────────────────────────────────

def generate_all_figures(
    metrics_df: pd.DataFrame,
    pred_store: dict | None = None,
    ts_store: dict | None = None,
    subgroup_metrics_df: pd.DataFrame | None = None,
    fig_dir: Path = FIGURE_DIR,
) -> None:
    """
    모든 Figure 생성 (run_experiment.py에서 호출).

    pred_store: {(model, mask_type, loss_rate): {var: (y_true, y_pred)}}
    ts_store  : {mask_type: {'gt': df, 'imputed': {model: df}, 'mask': df}}
    subgroup_metrics_df: season·metadata별 subgroup metric rows
    """
    print("\n[ Generating Figures ]")
    metrics_df = _filter_models_df(metrics_df)
    if subgroup_metrics_df is not None:
        subgroup_metrics_df = _filter_models_df(subgroup_metrics_df)

    # Figure 2: loss_rate 곡선
    for mask_type in metrics_df["mask_type"].unique():
        plot_loss_rate_curves(metrics_df, fig_dir=fig_dir, mask_type=mask_type)

    # Figure 1: scatter (열 순서·모델 집합은 results_all.csv와 동일한 metrics_df 기준)
    if pred_store:
        lr_fig1 = 0.30
        for mask_type in metrics_df["mask_type"].unique():
            all_model_preds = {
                k2[0]: v2
                for k2, v2 in pred_store.items()
                if k2[1] == mask_type and np.isclose(k2[2], lr_fig1, rtol=0, atol=1e-9)
            }
            if not all_model_preds:
                continue
            model_order = [m for m in MODEL_ORDER if m in all_model_preds]
            if not model_order:
                model_order = _fig1_model_order_from_metrics(metrics_df, mask_type, lr_fig1)
            plot_scatter(
                all_model_preds,
                fig_dir=fig_dir,
                mask_type=mask_type,
                loss_rate=lr_fig1,
                model_order=model_order,
            )

    # Figure 3: 시계열
    if ts_store:
        for mask_type, store in ts_store.items():
            meta = {k: store.get(k) for k in ("gh_name", "freq", "loss_rate", "test_ratio", "n_total")}
            for var in VARIABLES:
                plot_timeseries(
                    ground_truth=store["gt"],
                    imputed_dict=store["imputed"],
                    mask_matrix=store["mask"],
                    variable=var,
                    fig_dir=fig_dir,
                    suffix=mask_type,
                    scalers=store.get("scalers"),
                    meta=meta,
                )

    # Figure 4: Ablation — 결측률 민감도 분석
    plot_ablation_sensitivity(metrics_df, fig_dir=fig_dir)

    # Subgroup Figures: season / structure / covering / area
    if subgroup_metrics_df is not None and not subgroup_metrics_df.empty:
        for group_type in SUBGROUP_FIGURE_CONFIG:
            plot_subgroup_nmae(subgroup_metrics_df, group_type=group_type, fig_dir=fig_dir)

    print("  All figures generated")

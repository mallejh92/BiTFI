"""
evaluate.py
보간 모델 평가 모듈

평가 지표:
  MSE  — Mean Squared Error    (원래 스케일 역정규화 후 계산)
  MAE  — Mean Absolute Error   (원래 스케일 역정규화 후 계산)

훈련 손실: MSE (마스킹된 패치에 대해서만 계산, patchtst_model.py 참조)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def _normalized_mae(mae: float, norm_range: float | None, y_true: np.ndarray | None = None) -> float:
    """MAE를 변수 스케일로 나눠 단위 없는 지표를 만든다.

    norm_range(온실별 변수 전체 관측 range)가 주어지면 그것으로 나눈다. 이 값은
    gap/repeat/model에 독립인 고정 상수라서 gap 길이별 비교가 공정하다.

    norm_range가 없을 때만(하위호환) 평가 지점(y_true)의 max-min으로 나누는데,
    이 방식은 gap이 커질수록 분모가 커져 NMAE가 인위적으로 작아지는 편향이 있어
    권장하지 않는다(F4 gap-dependent bias 참조).
    """
    if norm_range is not None and np.isfinite(norm_range) and norm_range > 1e-8:
        return float(mae / norm_range)
    if y_true is None or len(y_true) == 0:
        return np.nan
    value_range = float(np.nanmax(y_true) - np.nanmin(y_true))
    if value_range <= 1e-8:
        return np.nan
    return float(mae / value_range)


# ──────────────────────────────────────────────
# 역정규화
# ──────────────────────────────────────────────

def _inverse_transform(
    df: pd.DataFrame,
    scalers: dict[str, MinMaxScaler | None],
) -> pd.DataFrame:
    # float64로 캐스팅하여 pandas 2.x LossySetitemError 방지
    df_inv = df.astype(np.float64)
    for col in df.columns:
        scaler = scalers.get(col)
        if scaler is None:
            continue
        not_null = df[col].notna()
        # 해당 열에 유효 샘플이 하나도 없으면 inverse_transform을 건너뛰어 빈 배열 예외를 방지한다.
        if not bool(not_null.any()):
            continue
        vals     = df.loc[not_null, col].values.astype(np.float64).reshape(-1, 1)
        inv_vals = scaler.inverse_transform(vals).flatten().astype(np.float64)
        assert inv_vals.ndim == 1, f"[{col}] inverse_transform produced {inv_vals.ndim}D"
        df_inv.loc[not_null, col] = inv_vals
    return df_inv


# ──────────────────────────────────────────────
# 메인 평가 함수
# ──────────────────────────────────────────────

def compute_metrics_v2(
    ground_truth: pd.DataFrame,
    imputed:      pd.DataFrame,
    eval_mask:    pd.DataFrame,
    scalers:      dict[str, MinMaxScaler | None] | None = None,
) -> dict[str, dict[str, float]]:
    """
    논문용 평가 함수 (v2). eval_mask를 직접 받는다.

    eval_mask 정의:
        True  = 인위 결측(artificial_mask == 0) AND 원본 유효(original_valid_mask == 1)
              → 평가 대상 위치
        False = 그 외 (평가하지 않음)

    Returns
    -------
    dict: variable → {MAE, RMSE, nRMSE, R2, n_eval}
    """
    if scalers is not None:
        gt_inv  = _inverse_transform(ground_truth, scalers)
        imp_inv = _inverse_transform(imputed,      scalers)
    else:
        gt_inv  = ground_truth.copy()
        imp_inv = imputed.copy()

    metrics: dict[str, dict[str, float]] = {}

    for col in ground_truth.columns:
        if col not in imputed.columns:
            continue
        if col not in eval_mask.columns:
            continue

        mask = eval_mask[col].values.astype(bool)
        y_true = np.asarray(gt_inv.loc[:, col].values,  dtype=np.float64).flatten()[mask]
        y_pred = np.asarray(imp_inv.loc[:, col].values, dtype=np.float64).flatten()[mask]

        valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true, y_pred = y_true[valid], y_pred[valid]

        if len(y_true) < 2:
            metrics[col] = {"MAE": np.nan, "RMSE": np.nan, "nRMSE": np.nan, "R2": np.nan, "n_eval": 0}
            continue

        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        val_range = float(np.nanmax(y_true) - np.nanmin(y_true))
        nrmse = float(rmse / val_range) if val_range > 1e-8 else np.nan

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

        metrics[col] = {"MAE": mae, "RMSE": rmse, "nRMSE": nrmse, "R2": r2, "n_eval": len(y_true)}

    return metrics


def compute_metrics(
    ground_truth:    pd.DataFrame,
    imputed:         pd.DataFrame,
    original_mask:   pd.DataFrame,
    artificial_mask: pd.DataFrame,
    scalers:         dict[str, MinMaxScaler | None] | None = None,
) -> dict[str, dict[str, float]]:
    """
    변수별 MSE, MAE 계산.

    평가 대상:
      인위 결측(artificial_mask == 0) & 원본 유효(original_mask == 1) 교집합.

    Parameters
    ----------
    ground_truth    : 원본(마스킹 전) 시계열 DataFrame
    imputed         : 모델이 보간한 시계열 DataFrame
    original_mask   : 원본 결측 위치 (1=유효, 0=원본 결측)
    artificial_mask : 인위 마스킹 위치 (1=유효, 0=인위 결측)
    scalers         : MinMaxScaler dict (None이면 역정규화 생략)
    """
    if scalers is not None:
        gt_inv  = _inverse_transform(ground_truth, scalers)
        imp_inv = _inverse_transform(imputed,      scalers)
    else:
        gt_inv  = ground_truth.copy()
        imp_inv = imputed.copy()

    metrics: dict[str, dict[str, float]] = {}

    for col in ground_truth.columns:
        if col not in imputed.columns:
            continue

        eval_mask = (artificial_mask[col] == 0) & (original_mask[col] == 1)
        y_true    = gt_inv.loc[eval_mask, col].values
        y_pred    = imp_inv.loc[eval_mask, col].values

        # ── Shape 정규화: 반드시 1-D float64 배열 ──
        y_true = np.asarray(y_true, dtype=np.float64).flatten()
        y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
        assert y_true.shape == y_pred.shape, (
            f"[{col}] shape mismatch: y_true={y_true.shape} vs y_pred={y_pred.shape}"
        )

        # NaN / Inf 제거
        valid  = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        if len(y_true) < 2:
            metrics[col] = {"MSE": np.nan, "MAE": np.nan, "NMAE": np.nan, "n_eval": 0}
            continue

        # MSE: 훈련 손실과 동일한 지표
        mse  = float(np.mean((y_true - y_pred) ** 2))
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(mse))
        nmae = _normalized_mae(mae, None, y_true)

        val_range = float(np.nanmax(y_true) - np.nanmin(y_true))
        nrmse = float(rmse / val_range) if val_range > 1e-8 else np.nan

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

        metrics[col] = {
            "MSE":    mse,
            "MAE":    mae,
            "RMSE":   rmse,
            "NMAE":   nmae,
            "nRMSE":  nrmse,
            "R2":     r2,
            "n_eval": len(y_true),
        }

    return metrics


def compute_metrics_by_subgroup(
    ground_truth:    pd.DataFrame,
    imputed:         pd.DataFrame,
    original_mask:   pd.DataFrame,
    artificial_mask: pd.DataFrame,
    group_labels:    dict[str, pd.Series],
    scalers:         dict[str, MinMaxScaler | None] | None = None,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """
    변수별 MSE, MAE를 season, structure_type 같은 그룹 라벨별로 나누어 계산.

    평가 대상:
      인위 결측(artificial_mask == 0) & 원본 유효(original_mask == 1) & 그룹 라벨 일치.
    """
    if scalers is not None:
        gt_inv  = _inverse_transform(ground_truth, scalers)
        imp_inv = _inverse_transform(imputed,      scalers)
    else:
        gt_inv  = ground_truth.copy()
        imp_inv = imputed.copy()

    subgroup_metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    # 각 그룹 라벨을 test index에 맞춰 정렬해 같은 시점끼리 비교되도록 한다.
    aligned_labels = {
        group_type: labels.reindex(ground_truth.index)
        for group_type, labels in group_labels.items()
    }

    for col in ground_truth.columns:
        if col not in imputed.columns:
            continue

        base_eval_mask = (artificial_mask[col] == 0) & (original_mask[col] == 1)

        for group_type, labels in aligned_labels.items():
            subgroup_metrics.setdefault(group_type, {})
            group_values = labels.dropna().astype(str).unique()

            for group_value in group_values:
                subgroup_metrics[group_type].setdefault(group_value, {})
                group_mask = labels.astype(str) == group_value
                eval_mask = base_eval_mask & group_mask

                y_true = gt_inv.loc[eval_mask, col].values
                y_pred = imp_inv.loc[eval_mask, col].values

                # 계산 안정성을 위해 평가 배열을 1차원 float64로 맞춘다.
                y_true = np.asarray(y_true, dtype=np.float64).flatten()
                y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
                assert y_true.shape == y_pred.shape, (
                    f"[{col}/{group_type}={group_value}] shape mismatch: "
                    f"y_true={y_true.shape} vs y_pred={y_pred.shape}"
                )

                # NaN / Inf 값을 제외하고 실제 비교 가능한 값만 지표에 사용한다.
                valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
                y_true = y_true[valid]
                y_pred = y_pred[valid]

                if len(y_true) < 2:
                    subgroup_metrics[group_type][group_value][col] = {
                        "MSE": np.nan,
                        "MAE": np.nan,
                        "NMAE": np.nan,
                        "n_eval": 0,
                    }
                    continue

                mse = float(np.mean((y_true - y_pred) ** 2))
                mae = float(np.mean(np.abs(y_true - y_pred)))
                nmae = _normalized_mae(mae, None, y_true)

                subgroup_metrics[group_type][group_value][col] = {
                    "MSE": mse,
                    "MAE": mae,
                    "NMAE": nmae,
                    "n_eval": len(y_true),
                }

    return subgroup_metrics


def compute_all_metrics(
    ground_truth:    pd.DataFrame,
    imputed:         pd.DataFrame,
    original_mask:   pd.DataFrame,
    artificial_mask: pd.DataFrame,
    group_labels:    dict[str, pd.Series],
    scalers:         dict[str, MinMaxScaler | None] | None = None,
    norm_ranges:     dict[str, float] | None = None,
) -> tuple[dict, dict]:
    """
    역정규화를 한 번만 수행하고 compute_metrics + compute_metrics_by_subgroup 결과를 동시에 반환.

    norm_ranges : {변수: 온실 전체 관측 range(원래 스케일)}. 주어지면 NMAE 분모로
                  이 고정값을 쓴다(gap/repeat 독립). 없으면 평가 지점 range로 fallback.
    """
    nr = norm_ranges or {}
    if scalers is not None:
        gt_inv  = _inverse_transform(ground_truth, scalers)
        imp_inv = _inverse_transform(imputed,      scalers)
    else:
        gt_inv  = ground_truth.copy()
        imp_inv = imputed.copy()

    # ── 변수별 지표 ──
    metrics: dict[str, dict[str, float]] = {}
    for col in ground_truth.columns:
        if col not in imputed.columns:
            continue
        eval_mask = (artificial_mask[col] == 0) & (original_mask[col] == 1)
        y_true = np.asarray(gt_inv.loc[eval_mask, col].values, dtype=np.float64).flatten()
        y_pred = np.asarray(imp_inv.loc[eval_mask, col].values, dtype=np.float64).flatten()
        assert y_true.shape == y_pred.shape, (
            f"[{col}] shape mismatch: y_true={y_true.shape} vs y_pred={y_pred.shape}"
        )
        valid  = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true, y_pred = y_true[valid], y_pred[valid]
        if len(y_true) < 2:
            metrics[col] = {"MSE": np.nan, "MAE": np.nan, "NMAE": np.nan, "n_eval": 0}
        else:
            mse = float(np.mean((y_true - y_pred) ** 2))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            metrics[col] = {"MSE": mse, "MAE": mae,
                            "NMAE": _normalized_mae(mae, nr.get(col), y_true), "n_eval": len(y_true)}

    # ── 서브그룹 지표 ──
    subgroup_metrics: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    aligned_labels = {
        group_type: labels.reindex(ground_truth.index)
        for group_type, labels in group_labels.items()
    }
    for col in ground_truth.columns:
        if col not in imputed.columns:
            continue
        base_eval_mask = (artificial_mask[col] == 0) & (original_mask[col] == 1)
        for group_type, labels in aligned_labels.items():
            subgroup_metrics.setdefault(group_type, {})
            for group_value in labels.dropna().astype(str).unique():
                subgroup_metrics[group_type].setdefault(group_value, {})
                eval_mask = base_eval_mask & (labels.astype(str) == group_value)
                y_true = np.asarray(gt_inv.loc[eval_mask, col].values, dtype=np.float64).flatten()
                y_pred = np.asarray(imp_inv.loc[eval_mask, col].values, dtype=np.float64).flatten()
                assert y_true.shape == y_pred.shape, (
                    f"[{col}/{group_type}={group_value}] shape mismatch: "
                    f"y_true={y_true.shape} vs y_pred={y_pred.shape}"
                )
                valid  = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
                y_true, y_pred = y_true[valid], y_pred[valid]
                if len(y_true) < 2:
                    subgroup_metrics[group_type][group_value][col] = {
                        "MSE": np.nan, "MAE": np.nan, "NMAE": np.nan, "n_eval": 0,
                    }
                else:
                    mse = float(np.mean((y_true - y_pred) ** 2))
                    mae = float(np.mean(np.abs(y_true - y_pred)))
                    subgroup_metrics[group_type][group_value][col] = {
                        "MSE": mse, "MAE": mae,
                        "NMAE": _normalized_mae(mae, nr.get(col), y_true),
                        "n_eval": len(y_true),
                    }

    return metrics, subgroup_metrics


# ──────────────────────────────────────────────
# 결과 집계 및 저장
# ──────────────────────────────────────────────

def evaluate_all(
    results_list: list[dict],
    save_path:    str | Path | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(results_list)
    cols_order = [
        "model", "greenhouse", "variable", "mask_type", "loss_rate",
        "MSE", "MAE", "NMAE", "n_eval",
        "context_len", "fit_seconds", "impute_seconds", "total_seconds",
        "model_key", "backbone",
    ]
    df = df[[c for c in cols_order if c in df.columns]]
    df = df.sort_values(["model", "mask_type", "loss_rate", "variable"])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n결과 저장: {save_path}")

    return df


def evaluate_subgroup_all(
    results_list: list[dict],
    save_path:    str | Path | None = None,
) -> pd.DataFrame:
    # 서브그룹 결과는 group_type과 group_value를 보존한 상태로 정렬해 저장한다.
    df = pd.DataFrame(results_list)
    cols_order = [
        "model", "greenhouse", "variable", "mask_type", "loss_rate",
        "group_type", "group_value", "MSE", "MAE", "NMAE", "n_eval",
        "context_len",
    ]
    df = df[[c for c in cols_order if c in df.columns]]
    df = df.sort_values(["group_type", "group_value", "model", "mask_type", "loss_rate", "variable"])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"\n서브그룹 결과 저장: {save_path}")

    return df


def print_summary(df: pd.DataFrame) -> None:
    """모델별 × 마스킹 타입별 평균 MSE / MAE 출력"""
    print("\n" + "=" * 70)
    print("모델별 평균 성능 (전 변수 평균)")
    print("=" * 70)
    metric_cols = [c for c in ["MSE", "MAE"] if c in df.columns]
    summary = (
        df.groupby(["model", "mask_type"])[metric_cols]
        .mean()
        .round(5)
    )
    print(summary.to_string())
    print("=" * 70)


def print_subgroup_summary(df: pd.DataFrame) -> None:
    """서브그룹별 평균 NMAE와 평가 샘플 수를 콘솔에 출력"""
    print("\n" + "=" * 70)
    print("서브그룹별 평균 성능 (NMAE, 전 변수 평균)")
    print("=" * 70)
    if df.empty:
        print("서브그룹 결과가 없습니다.")
        print("=" * 70)
        return

    summary = (
        df.groupby(["group_type", "group_value", "model", "mask_type"])
        .agg(NMAE=("NMAE", "mean"), n_eval=("n_eval", "sum"))
        .round({"NMAE": 5})
        .reset_index()
    )
    print(summary.to_string(index=False))
    print("=" * 70)


def print_results_table(df: pd.DataFrame) -> None:
    """
    모델별 × 마스킹 타입별 MSE, MAE 비교 테이블을 Markdown 형식으로 출력.

    출력 구조
    ---------
    1. 전체 평균 (전 변수 × 전 loss_rate 평균)
    2. 변수별 상세 (전 loss_rate 평균)
    """
    metric_cols = [c for c in ["MSE", "MAE"] if c in df.columns]
    if not metric_cols:
        print("MSE / MAE 컬럼을 찾을 수 없습니다.")
        return

    # ── 1. 전체 평균 테이블 ──
    print("\n" + "=" * 80)
    # cp949 콘솔 호환성을 위해 헤더 문자열은 ASCII 구분자를 사용한다.
    print("[ 모델별 성능 비교 - 전 변수 x 전 loss_rate 평균 ]")
    print("=" * 80)

    summary = (
        df.groupby(["model", "mask_type"])[metric_cols]
        .mean()
        .round(5)
        .reset_index()
    )

    col_w_model = max(summary["model"].str.len().max(), 6) + 2
    col_w_mask  = max(summary["mask_type"].str.len().max(), 8) + 2
    metric_w    = 12

    header = (
        f"| {'모델':<{col_w_model}} | {'마스킹':<{col_w_mask}} | "
        + " | ".join(f"{c:>{metric_w}}" for c in metric_cols)
        + " |"
    )
    sep = (
        "|" + "-" * (col_w_model + 2)
        + "|" + "-" * (col_w_mask + 2)
        + "|" + "|".join(["-" * (metric_w + 2)] * len(metric_cols))
        + "|"
    )

    print(header)
    print(sep)
    for _, row in summary.iterrows():
        vals = " | ".join(f"{row[c]:>{metric_w}.5f}" for c in metric_cols)
        print(f"| {row['model']:<{col_w_model}} | {row['mask_type']:<{col_w_mask}} | {vals} |")

    # ── 2. 변수별 상세 테이블 ──
    print("\n[ 변수별 상세 - 전 loss_rate 평균 ]")
    print(sep[:len(sep)])  # 재사용

    detail = (
        df.groupby(["model", "mask_type", "variable"])[metric_cols]
        .mean()
        .round(5)
        .reset_index()
    )

    header2 = (
        f"| {'모델':<{col_w_model}} | {'마스킹':<{col_w_mask}} | {'변수':^4} | "
        + " | ".join(f"{c:>{metric_w}}" for c in metric_cols)
        + " |"
    )
    sep2 = (
        "|" + "-" * (col_w_model + 2)
        + "|" + "-" * (col_w_mask + 2)
        + "|------"
        + "|" + "|".join(["-" * (metric_w + 2)] * len(metric_cols))
        + "|"
    )
    print(header2)
    print(sep2)
    for _, row in detail.iterrows():
        vals = " | ".join(f"{row[c]:>{metric_w}.5f}" for c in metric_cols)
        print(
            f"| {row['model']:<{col_w_model}} | {row['mask_type']:<{col_w_mask}} "
            f"| {row['variable']:^4} | {vals} |"
        )

    print("=" * 80)

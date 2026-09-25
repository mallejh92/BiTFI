"""
fig_comparison.py
교차-온실 비교 실험(run_comparison.py)의 계절×시간대 층화 시각화 (개선사항.md #4).

입력 : 03_result/comparison/results.csv
        (컬럼: greenhouse, model, scenario, masked_vars, gap_length_h, repeat,
               group_type, group_value, variable, MSE, MAE, NMAE, n_eval, context_len)
출력 : 03_result/comparison/figures/
        1. Fig_comparison_panels  — 2×2(여름/겨울 × 오전/오후) 변수별 모델 MAE 막대
        2. Fig_comparison_heatmap — 모델 × (계절×시간대) NMAE 히트맵
        3. (옵션) Fig_comparison_structure — meta_data.xlsx 구조별 서브그룹 비교

fig_utils의 스타일/팔레트/저장 헬퍼를 재사용한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fig_utils as fu  # noqa: E402

RESULT_DIR = _THIS_DIR / ".." / ".." / "03_result" / "comparison"
VARIABLES = fu.VARIABLES  # ['Tin','Tout','RH','CO2','Rad']

# 2×2 패널 구성 (group_type='season_tod' 값)
PANEL_CELLS = [
    ("summer_AM", "Summer · AM"),
    ("summer_PM", "Summer · PM"),
    ("winter_AM", "Winter · AM"),
    ("winter_PM", "Winter · PM"),
]


def _model_color(model: str) -> str:
    """fig_utils 팔레트 재사용(없으면 회색). AG-* 는 RecursiveTabular 색 계열로 묶는다."""
    if model in fu.MODEL_COLORS:
        return fu.MODEL_COLORS[model]
    if model.startswith("AG-"):
        return fu.MODEL_COLORS.get("RecursiveTabular", "#0072B2")
    if model.startswith("TimesFM"):
        return fu.MODEL_COLORS.get("TimesFM", "#009E73")
    return "#999999"


def _ordered_models(df: pd.DataFrame) -> list[str]:
    models = list(df["model"].unique())
    # 복잡도 오름차순 우선 정렬, 미정의는 알파벳순으로 뒤에.
    pref = ["LI", "SeasonalNaive", "AG-LightGBM", "AG-RandomForest",
            "AG-DeepAR", "AG-PatchTST", "Chronos2", "TimesFM2.5", "CAFI"]
    head = [m for m in pref if m in models]
    tail = sorted(m for m in models if m not in pref)
    return head + tail


def plot_panels(df: pd.DataFrame, out_dir: Path) -> None:
    """2×2(계절×시간대) × 변수별 모델 MAE 막대."""
    sub = df[df["group_type"] == "season_tod"]
    if sub.empty:
        print("[fig_comparison] season_tod 그룹이 없어 패널을 건너뜁니다.")
        return
    models = _ordered_models(sub)
    x = np.arange(len(VARIABLES))
    width = 0.8 / max(len(models), 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
    for ax, (cell, title) in zip(axes.flat, PANEL_CELLS):
        cell_df = sub[sub["group_value"] == cell]
        for mi, model in enumerate(models):
            md = cell_df[cell_df["model"] == model]
            means = [md[md["variable"] == v]["MAE"].mean() for v in VARIABLES]
            ax.bar(x + mi * width, means, width, label=model, color=_model_color(model))
        ax.set_title(title)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(VARIABLES)
        ax.set_ylabel("MAE")
    # 범례는 그림 상단에 한 번만
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(models), 5),
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Imputation MAE by Season × Time-of-day", y=1.06, fontsize=14)
    fig.tight_layout()
    fu.save_figure(fig, out_dir, "Fig_comparison_panels")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """모델 × (계절×시간대) 평균 NMAE 히트맵."""
    sub = df[df["group_type"] == "season_tod"]
    if sub.empty:
        print("[fig_comparison] season_tod 그룹이 없어 히트맵을 건너뜁니다.")
        return
    models = _ordered_models(sub)
    cells = [c for c, _ in PANEL_CELLS if c in set(sub["group_value"])]
    mat = np.full((len(models), len(cells)), np.nan)
    for i, model in enumerate(models):
        for j, cell in enumerate(cells):
            vals = sub[(sub["model"] == model) & (sub["group_value"] == cell)]["NMAE"]
            mat[i, j] = vals.mean()

    fig, ax = plt.subplots(figsize=(1.6 * len(cells) + 2, 0.55 * len(models) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([dict(PANEL_CELLS).get(c, c) for c in cells], rotation=20, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(cells)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        color="white" if mat[i, j] > np.nanmean(mat) else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="NMAE (mean)")
    ax.set_title("NMAE Heatmap (Model × Season×Time-of-day)")
    fig.tight_layout()
    fu.save_figure(fig, out_dir, "Fig_comparison_heatmap")
    plt.close(fig)


def plot_structure(df: pd.DataFrame, out_dir: Path, meta_path: Path | None = None) -> None:
    """(옵션) meta_data.xlsx의 구조(Multi/Single-span)별 모델 NMAE 비교."""
    if meta_path is None or not Path(meta_path).exists():
        print("[fig_comparison] meta_data.xlsx 없음 → 구조 비교 생략(옵션).")
        return
    try:
        meta = pd.read_excel(meta_path, dtype=str)
    except Exception as e:
        print(f"[fig_comparison] meta 로드 실패 → 구조 비교 생략: {e}")
        return
    # greenhouse → structure 매핑 (컬럼명 추정: 'type' 또는 'structure' 포함)
    id_col = next((c for c in meta.columns if "id" in c.lower() or "PF" in str(meta[c].iloc[0])), meta.columns[0])
    struct_col = next((c for c in meta.columns if "type" in c.lower() or "span" in c.lower() or "구조" in c), None)
    if struct_col is None:
        print("[fig_comparison] 구조 컬럼을 찾지 못함 → 생략.")
        return
    mp = dict(zip(meta[id_col].astype(str), meta[struct_col].astype(str)))
    all_df = df[df["group_type"] == "all"].copy()
    all_df["structure"] = all_df["greenhouse"].astype(str).map(mp)
    all_df = all_df.dropna(subset=["structure"])
    if all_df.empty:
        print("[fig_comparison] 구조 매핑 결과 없음 → 생략.")
        return
    models = _ordered_models(all_df)
    structs = sorted(all_df["structure"].unique())
    x = np.arange(len(structs))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(2 * len(structs) + 3, 5))
    for mi, model in enumerate(models):
        means = [all_df[(all_df["model"] == model) & (all_df["structure"] == s)]["NMAE"].mean()
                 for s in structs]
        ax.bar(x + mi * width, means, width, label=model, color=_model_color(model))
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(structs)
    ax.set_ylabel("NMAE")
    ax.set_title("NMAE by Greenhouse Structure")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fu.save_figure(fig, out_dir, "Fig_comparison_structure")
    plt.close(fig)


def generate_all(result_dir: Path | None = None, meta_path: Path | None = None) -> None:
    fu.setup_figure_style()
    result_dir = Path(result_dir) if result_dir else RESULT_DIR
    csv_path = result_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv 없음: {csv_path}. 먼저 run_comparison.py 실행.")
    df = pd.read_csv(csv_path)
    out_dir = result_dir / "figures"
    print(f"[fig_comparison] {len(df)}행 로드 → {out_dir}")
    plot_panels(df, out_dir)
    plot_heatmap(df, out_dir)
    plot_structure(df, out_dir, meta_path=meta_path)
    print("[fig_comparison] 완료.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="교차-온실 비교 층화 시각화")
    p.add_argument("--result-dir", type=str, default=str(RESULT_DIR))
    p.add_argument("--meta", type=str, default="", help="meta_data.xlsx 경로(옵션)")
    args = p.parse_args()
    generate_all(
        result_dir=Path(args.result_dir),
        meta_path=Path(args.meta) if args.meta else None,
    )

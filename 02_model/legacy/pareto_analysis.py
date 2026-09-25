"""
pareto_analysis.py
파레토 프런트 분석

입력: 03_result/context_ablation/ablation_results.csv
출력: 파레토 프런트 데이터 CSV + 논문용 Figure 4 (04_figure/fig4_pareto_front.pdf)

파레토 프런트 정의:
  x축: impute_seconds (추론 시간, 효율성)
  y축: MAE (정확도)
  파레토 최적: 같은 시간에 더 낮은 MAE, 또는 같은 MAE에 더 짧은 시간

원예학적 의의:
  - 정확도와 연산 비용의 최적 트레이드오프 지점을 찾는다
  - 실제 온실 모니터링 시스템 도입 시 비용-효과 기준이 된다
  - 파레토 프런트에 있는 (context_len, model) 조합이 권고 설정이다

실행:
  python pareto_analysis.py [--input 03_result/context_ablation/ablation_results.csv]
                             [--output-dir 04_figure]
                             [--metric MAE]  # MAE or MSE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# legacy/ 로 이동했으므로 02_model/(부모)까지 sys.path에 포함해야
# preprocessing·evaluate·masking·gpu_utils·models 패키지를 찾을 수 있다.
_SCRIPT_DIR = Path(__file__).resolve().parent
_THIS_DIR = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import gpu_utils  # noqa: F401, E402

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless 환경에서도 동작
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────
# Publication-quality 설정
# ──────────────────────────────────────────────
FIG_SIZE = (8, 6)
DPI = 300
MARKER_SIZE = 80

COLOR_PALETTE = {
    "CAFI": "#0072B2",
    "AutoGluon": "#E69F00",
    "TimesFM": "#009E73",
}
DEFAULT_COLORS = ["#56B4E9", "#D55E00", "#CC79A7", "#F0E442", "#999999"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _apply_rc():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
    })


# ──────────────────────────────────────────────
# 데이터 로드 및 집계
# ──────────────────────────────────────────────
def load_results(input_path: Path) -> pd.DataFrame | None:
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"[ERROR] 입력 파일이 없습니다: {input_path}")
        print("  먼저 run_context_ablation.py 를 실행해 ablation_results.csv 를 생성하세요.")
        return None
    try:
        df = pd.read_csv(input_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"[ERROR] 입력 파일 로드 실패: {e}")
        return None
    if df.empty:
        print("[ERROR] 입력 파일이 비어 있습니다.")
        return None
    return df


def aggregate(df: pd.DataFrame, metric: str = "MAE") -> pd.DataFrame:
    """
    (model, context_len) 수준으로 mean metric, mean impute_seconds 집계.
    온실·변수 전체 평균.
    """
    required = {"model", "context_len", metric, "impute_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    agg = (
        df.groupby(["model", "context_len"], as_index=False)
        .agg(
            **{
                metric: (metric, "mean"),
                "impute_seconds": ("impute_seconds", "mean"),
            }
        )
        .sort_values(["model", "context_len"])
        .reset_index(drop=True)
    )
    # NaN metric 제거 (평가 불가 조합)
    agg = agg.dropna(subset=[metric, "impute_seconds"]).reset_index(drop=True)
    return agg


# ──────────────────────────────────────────────
# 파레토 프런트
# ──────────────────────────────────────────────
def _pareto_flags(times: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """
    minimization 파레토: x=time, y=error.
    한 점이 다른 점보다 time과 error가 모두 (한쪽은 strictly) 낮거나 같으면 dominated.
    """
    n = len(times)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j가 i를 지배: j가 둘 다 <= 이고 적어도 하나는 strictly <
            if (times[j] <= times[i] and errors[j] <= errors[i] and
                    (times[j] < times[i] or errors[j] < errors[i])):
                is_pareto[i] = False
                break
    return is_pareto


def compute_pareto(agg: pd.DataFrame, metric: str = "MAE") -> pd.DataFrame:
    """
    각 모델별로 파레토 최적 여부를 표시한 DataFrame 반환.
    columns: model, context_len, <metric>, impute_seconds, is_pareto
    """
    out_frames = []
    for model, g in agg.groupby("model"):
        g = g.copy().reset_index(drop=True)
        times = g["impute_seconds"].values.astype(float)
        errors = g[metric].values.astype(float)
        g["is_pareto"] = _pareto_flags(times, errors)
        out_frames.append(g)

    result = pd.concat(out_frames, ignore_index=True)
    result = result[["model", "context_len", metric, "impute_seconds", "is_pareto"]]
    return result.sort_values(["model", "impute_seconds"]).reset_index(drop=True)


def overall_pareto(agg: pd.DataFrame, metric: str = "MAE") -> pd.DataFrame:
    """모든 모델을 합친 전체 파레토 프런트 점들 (시간 오름차순)."""
    times = agg["impute_seconds"].values.astype(float)
    errors = agg[metric].values.astype(float)
    flags = _pareto_flags(times, errors)
    front = agg[flags].copy()
    return front.sort_values("impute_seconds").reset_index(drop=True)


def _pareto_step(front: pd.DataFrame, metric: str):
    """
    파레토 점들을 시간 오름차순으로 정렬하고 error가 단조 감소하는
    step-function 좌표를 만든다.
    """
    f = front.sort_values("impute_seconds").reset_index(drop=True)
    xs = f["impute_seconds"].values.astype(float)
    ys = f[metric].values.astype(float)
    # 시간 증가에 따라 error 최솟값만 유지 (lower envelope)
    keep_x, keep_y = [], []
    best = np.inf
    for x, y in zip(xs, ys):
        if y < best - 1e-12:
            keep_x.append(x)
            keep_y.append(y)
            best = y
    return np.array(keep_x), np.array(keep_y)


# ──────────────────────────────────────────────
# 권고 구성 (knee point)
# ──────────────────────────────────────────────
def recommend_config(pareto_df: pd.DataFrame, metric: str = "MAE") -> dict | None:
    """
    전체 파레토 점들 중 이상점 [min_time, min_metric]에 정규화 L2 거리가
    가장 가까운 knee point를 권고 구성으로 선택한다.
    """
    pts = pareto_df[pareto_df["is_pareto"]].copy()
    if pts.empty:
        pts = pareto_df.copy()
    if pts.empty:
        return None

    t = pts["impute_seconds"].values.astype(float)
    e = pts[metric].values.astype(float)

    t_min, t_max = t.min(), t.max()
    e_min, e_max = e.min(), e.max()
    t_rng = (t_max - t_min) or 1.0
    e_rng = (e_max - e_min) or 1.0

    t_norm = (t - t_min) / t_rng
    e_norm = (e - e_min) / e_rng

    # 이상점은 정규화 좌표계의 원점 [0, 0] (min_time, min_metric)
    dist = np.sqrt(t_norm ** 2 + e_norm ** 2)
    idx = int(np.argmin(dist))
    row = pts.iloc[idx]

    return {
        "model": row["model"],
        "context_len": int(row["context_len"]),
        metric: float(row[metric]),
        "impute_seconds": float(row["impute_seconds"]),
        "knee_distance": float(dist[idx]),
    }


# ──────────────────────────────────────────────
# Figure 4
# ──────────────────────────────────────────────
def make_figure(
    agg: pd.DataFrame,
    pareto_df: pd.DataFrame,
    overall_front: pd.DataFrame,
    recommended: dict | None,
    output_dir: Path,
    metric: str = "MAE",
) -> None:
    _apply_rc()
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    models = list(agg["model"].unique())
    color_iter = iter(DEFAULT_COLORS)
    model_color = {}
    model_marker = {}
    for i, m in enumerate(models):
        model_color[m] = COLOR_PALETTE.get(m, next(color_iter, "#444444"))
        model_marker[m] = MARKERS[i % len(MARKERS)]

    # ── 모델별 산점도 + 주석 + 모델별 파레토 step ──
    for m in models:
        g = agg[agg["model"] == m].sort_values("impute_seconds")
        color = model_color[m]
        marker = model_marker[m]

        ax.scatter(
            g["impute_seconds"], g[metric],
            s=MARKER_SIZE, color=color, marker=marker,
            edgecolors="black", linewidths=0.6,
            label=m, zorder=3,
        )

        # context_len 주석
        for _, row in g.iterrows():
            ax.annotate(
                f"{int(row['context_len'])}h",
                (row["impute_seconds"], row[metric]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=8, color=color, zorder=4,
            )

        # 모델별 파레토 프런트 step line
        m_front = pareto_df[(pareto_df["model"] == m) & (pareto_df["is_pareto"])]
        if len(m_front) >= 1:
            sx, sy = _pareto_step(m_front, metric)
            if len(sx) >= 2:
                ax.step(sx, sy, where="post", color=color,
                        linewidth=2, alpha=0.7, zorder=2)

    # ── 전체 파레토 프런트 (bold dashed) ──
    if len(overall_front) >= 1:
        ox, oy = _pareto_step(overall_front, metric)
        if len(ox) >= 2:
            ax.step(ox, oy, where="post", color="black",
                    linewidth=2.5, linestyle="--",
                    label="Overall Pareto Front", zorder=5)

    # ── 권고 구성 강조 ──
    if recommended is not None:
        ax.scatter(
            [recommended["impute_seconds"]], [recommended[metric]],
            s=MARKER_SIZE * 3, facecolors="none", edgecolors="red",
            linewidths=2.0, zorder=6,
            label=f"Recommended ({recommended['model']} "
                  f"{recommended['context_len']}h)",
        )

    ax.set_xlabel("Inference Time (seconds)")
    if metric == "MAE":
        ax.set_ylabel("MAE (°C equivalent)")
    else:
        ax.set_ylabel(metric)
    ax.set_title("Accuracy vs. Inference Time Pareto Front")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "fig4_pareto_front.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 저장: {pdf_path}")


# ──────────────────────────────────────────────
# 출력
# ──────────────────────────────────────────────
def print_summary(pareto_df: pd.DataFrame, recommended: dict | None, metric: str) -> None:
    print("\n" + "=" * 70)
    print("[ 파레토 분석 요약 ]")
    print("=" * 70)
    pareto_points = pareto_df[pareto_df["is_pareto"]].sort_values("impute_seconds")
    print(f"파레토 최적 점 수: {len(pareto_points)} / {len(pareto_df)}")
    print("\n파레토 최적 (model, context_len, {0}, impute_seconds):".format(metric))
    for _, r in pareto_points.iterrows():
        print(f"  {r['model']:<10} | {int(r['context_len']):>4}h | "
              f"{metric}={r[metric]:.5f} | impute={r['impute_seconds']:.2f}s")

    print("\n" + "-" * 70)
    if recommended is not None:
        print("권고 구성 (knee point — 정확도/시간 최적 균형):")
        print(f"  model        = {recommended['model']}")
        print(f"  context_len  = {recommended['context_len']}h")
        print(f"  {metric:<12} = {recommended[metric]:.5f}")
        print(f"  impute_secs  = {recommended['impute_seconds']:.2f}s")
    else:
        print("권고 구성을 산출할 수 없습니다.")
    print("=" * 70)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def run(input_path: Path, output_dir: Path, metric: str = "MAE") -> int:
    df = load_results(input_path)
    if df is None:
        return 1

    if metric not in df.columns:
        print(f"[ERROR] metric '{metric}' 컬럼이 입력에 없습니다. (가능: MAE, MSE)")
        return 1

    agg = aggregate(df, metric=metric)
    if agg.empty:
        print("[ERROR] 집계 결과가 비어 있습니다 (유효한 metric 값 없음).")
        return 1

    pareto_df = compute_pareto(agg, metric=metric)
    overall_front = overall_pareto(agg, metric=metric)
    recommended = recommend_config(pareto_df, metric=metric)

    # ── 파레토 결과 저장 ──
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pareto_path = output_dir / "pareto_results.csv"

    pareto_save = pareto_df.copy()
    pareto_save["is_overall_pareto"] = pareto_save.apply(
        lambda r: bool(
            ((overall_front["model"] == r["model"]) &
             (overall_front["context_len"] == r["context_len"])).any()
        ),
        axis=1,
    )
    if recommended is not None:
        pareto_save["is_recommended"] = pareto_save.apply(
            lambda r: (r["model"] == recommended["model"]
                       and int(r["context_len"]) == recommended["context_len"]),
            axis=1,
        )
    pareto_save.to_csv(pareto_path, index=False, encoding="utf-8-sig")
    print(f"파레토 결과 저장: {pareto_path}  ({len(pareto_save)} rows)")

    # ── Figure ──
    make_figure(agg, pareto_df, overall_front, recommended, output_dir, metric=metric)

    # ── 출력 ──
    print_summary(pareto_df, recommended, metric)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="파레토 프런트 분석 및 Figure 4 생성")
    p.add_argument("--input", default="03_result/context_ablation/ablation_results.csv",
                   help="ablation_results.csv 경로")
    p.add_argument("--output-dir", default="04_figure",
                   help="파레토 CSV / Figure 저장 디렉터리")
    p.add_argument("--metric", default="MAE", choices=["MAE", "MSE"],
                   help="파레토 y축 지표")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    inp = Path(args.input)
    if not inp.is_absolute():
        inp = (_THIS_DIR / inp).resolve()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = (_THIS_DIR / out).resolve()
    return run(inp, out, metric=args.metric)


if __name__ == "__main__":
    raise SystemExit(main())

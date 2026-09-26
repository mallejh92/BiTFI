"""
fig_utils.py
온실 환경 시계열 + 보간 논문용 그림 생성 공용 유틸리티.

- 출판 품질 rcParams 설정
- 컬러블라인드 친화 팔레트 (변수/모델)
- 변수 라벨/단위, loss rate, context length 정의
- 경로 헬퍼 (데이터 / 결과 / 그림 디렉토리)
- PDF + PNG 동시 저장 헬퍼

모든 그림 스크립트는 다음과 같이 import 한다::

    import fig_utils as fu
    fu.setup_figure_style()
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# 헤드리스 환경에서도 저장 가능하도록 Agg 백엔드 강제
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (Agg 백엔드 설정 후 import)


# ---------------------------------------------------------------------------
# 변수 라벨 / 단위
# ---------------------------------------------------------------------------
VARIABLE_LABELS = {
    "Tin":  r"Internal Temp. ($T_{in}$, °C)",
    "Tout": r"External Temp. ($T_{out}$, °C)",
    "RH":   "Rel. Humidity (RH, %)",
    "CO2":  r"CO$_2$ (ppm)",
    "Rad":  "Solar Radiation (Rad, W/m²)",
}

VARIABLE_UNITS = {
    "Tin":  "°C",
    "Tout": "°C",
    "RH":   "%",
    "CO2":  "ppm",
    "Rad":  "W/m²",
}

# 축 눈금/범례처럼 짧은 변수 라벨이 필요한 곳에서 공통으로 쓰는 표기(CO2 아래첨자 포함).
VAR_SHORT_LABELS = {
    "Tin":  r"$T_{in}$",
    "Tout": r"$T_{out}$",
    "RH":   "RH",
    "CO2":  r"CO$_2$",
    "Rad":  "Rad",
}

# 변수 출력 순서 (그림 전체에서 일관되게 사용)
VARIABLES = ["Tin", "Tout", "RH", "CO2", "Rad"]


# ---------------------------------------------------------------------------
# 컬러블라인드 친화 팔레트
# ---------------------------------------------------------------------------
VARIABLE_COLORS = {
    "Tin":  "#E69F00",
    "Tout": "#56B4E9",
    "RH":   "#009E73",
    "CO2":  "#CC79A7",
    "Rad":  "#F0E442",
}

MODEL_COLORS = {
    # Classical baselines
    "LinearInterp":    "#999999",
    "SeasonalNaive":   "#E69F00",
    # ML
    "RecursiveTabular":"#0072B2",
    # DL
    "PatchTST":        "#D55E00",
    # Foundation models
    "Chronos2":        "#56B4E9",
    # Proposed
    "CAFI":            "#CC79A7",
    # Legacy (이전 실험 결과 호환)
    "LSTM":            "#009E73",
    "TimesFM":         "#009E73",
    "AutoGluon":       "#E69F00",
}

MODEL_MARKERS = {
    "LinearInterp":    "x",
    "SeasonalNaive":   "^",
    "RecursiveTabular":"D",
    "PatchTST":        "P",
    "Chronos2":        "v",
    "CAFI":            "s",
    "LSTM":            "D",
    "TimesFM":         "^",
    "AutoGluon":       "o",
}

# 논문 비교 모델 순서 (복잡도 오름차순)
MODEL_ORDER = [
    "LinearInterp",
    "SeasonalNaive",
    "RecursiveTabular",
    "PatchTST",
    "Chronos2",
    "CAFI",
    # 레거시
    "LSTM",
    "TimesFM",
    "AutoGluon",
]

# 모델 범례용 표시 이름
MODEL_DISPLAY = {
    "LinearInterp":    "Linear Interp.",
    "SeasonalNaive":   "Seasonal Naive",
    "RecursiveTabular":"RecursiveTabular",
    "PatchTST":        "PatchTST",
    "Chronos2":        "Chronos-2",
    "CAFI":            "CAFI",
}


# ---------------------------------------------------------------------------
# 현재 비교 파이프라인(run_comparison.py)의 9개 모델 레지스트리
#   (results.csv의 'model' 컬럼 값과 일치)
# ---------------------------------------------------------------------------
# 키는 results.csv의 'model' 값(AG-*)과 반드시 일치해야 매칭된다.
# 표시명(범례/축)만 "AG-" 접두어를 떼어 LightGBM/Random Forest/LSTM/PatchTST로 보여준다.
CMP_MODEL_ORDER = [
    "LI", "SeasonalNaive",
    "AG-LightGBM", "AG-RandomForest", "AG-DeepAR", "AG-PatchTST",
    "Chronos2", "TimesFM2.5", "TimesFM3.0",
    "TimesFM3.0-MV", "TimesFM3.0-COV",
    "CAFI", "CAFI-TimesFM3",
    "Spatial-Ridge", "TimesFM3.0-COV-SPA",
    "CAFI-R1", "CAFI-TimesFM3-R1",
    "BiTFI-TimesFM3-fwd", "BiTFI-TimesFM3", "BiTFI-Chronos2",
]

CMP_MODEL_DISPLAY = {
    "LI":              "Linear Interp.",
    "SeasonalNaive":   "Seasonal Naive",
    "AG-LightGBM":     "LightGBM",
    "AG-RandomForest": "Random Forest",
    "AG-DeepAR":       "LSTM",
    "AG-PatchTST":     "PatchTST",
    "Chronos2":        "Chronos-2",
    "TimesFM2.5":      "TimesFM-2.5",
    "TimesFM3.0":      "TimesFM-3.0",
    "TimesFM3.0-MV":   "TimesFM-3.0 (MV)",
    "TimesFM3.0-COV":  "TimesFM-3.0 (cov)",
    "CAFI":            "CAFI (Chronos-2)",
    "CAFI-TimesFM3":   "CAFI (TimesFM-3.0)",
    "Spatial-Ridge":   "Spatial-Ridge",
    "TimesFM3.0-COV-SPA": "TimesFM-3.0 (cov+spa)",
    "CAFI-R1":         "CAFI-1 (Chronos-2)",
    "CAFI-TimesFM3-R1": "CAFI-1 (TimesFM-3.0)",
    "BiTFI-TimesFM3":   "BiTFI (TimesFM-3.0)",
    "BiTFI-Chronos2":   "BiTFI (Chronos-2)",
    "BiTFI-TimesFM3-fwd": "BiTFI-fwd (TimesFM-3.0)",
}

# 컬러블라인드 친화(Tol/Okabe-Ito 혼합), 10개 구분. CAFI는 제안기법이라 강조색.
CMP_MODEL_COLORS = {
    "LI":              "#999999",
    "SeasonalNaive":   "#DDCC77",
    "AG-LightGBM":     "#0072B2",
    "AG-RandomForest": "#88CCEE",
    "AG-DeepAR":       "#117733",
    "AG-PatchTST":     "#D55E00",
    "Chronos2":        "#332288",
    "TimesFM2.5":      "#44AA99",
    "TimesFM3.0":      "#882255",
    "TimesFM3.0-MV":   "#AA4499",
    "TimesFM3.0-COV":  "#661100",
    "CAFI":            "#CC79A7",
    "CAFI-TimesFM3":   "#E69F00",
    "Spatial-Ridge":   "#6699CC",
    "TimesFM3.0-COV-SPA": "#004488",
    "CAFI-R1":         "#AA3377",
    "CAFI-TimesFM3-R1": "#EE7733",
    "BiTFI-TimesFM3":   "#117733",
    "BiTFI-Chronos2":   "#44AA99",
    "BiTFI-TimesFM3-fwd": "#999933",
}

CMP_MODEL_MARKERS = {
    "LI": "x", "SeasonalNaive": "^",
    "AG-LightGBM": "D", "AG-RandomForest": "d", "AG-DeepAR": "P", "AG-PatchTST": "X",
    "Chronos2": "v", "TimesFM2.5": "o", "TimesFM3.0": "*", "TimesFM3.0-MV": "p", "TimesFM3.0-COV": "h", "CAFI": "s", "CAFI-TimesFM3": "*", "Spatial-Ridge": "1", "TimesFM3.0-COV-SPA": "8", "CAFI-R1": "D", "CAFI-TimesFM3-R1": "d", "BiTFI-TimesFM3": "P", "BiTFI-Chronos2": "X", "BiTFI-TimesFM3-fwd": "+",
}


def cmp_model_color(model: str) -> str:
    return CMP_MODEL_COLORS.get(model, "#333333")


def cmp_model_marker(model: str) -> str:
    return CMP_MODEL_MARKERS.get(model, "o")


def cmp_model_display(model: str) -> str:
    return CMP_MODEL_DISPLAY.get(model, model)


def cmp_lw(model: str) -> float:
    """제안기법(CAFI)은 더 두껍게 그려 강조한다."""
    return 2.6 if model == "CAFI" else 1.4


def ordered_models(present) -> list[str]:
    """results.csv에 존재하는 모델을 표준 순서로 정렬(미등록은 뒤에)."""
    present = list(present)
    head = [m for m in CMP_MODEL_ORDER if m in present]
    tail = [m for m in present if m not in CMP_MODEL_ORDER]
    return head + tail


# ---------------------------------------------------------------------------
# results.csv 로딩 / 집계 헬퍼
# ---------------------------------------------------------------------------
def get_comparison_dir() -> Path:
    """교차-온실 비교 결과 디렉토리(03_result/comparison)."""
    return get_result_dir("comparison")


def load_results(result_dir: str | Path | None = None) -> "pd.DataFrame":
    """results.csv를 읽어 DataFrame으로 반환."""
    rd = Path(result_dir) if result_dir else get_comparison_dir()
    csv_path = rd / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv 없음: {csv_path}. 먼저 run_comparison.py 실행.")
    return pd.read_csv(csv_path)


def load_greenhouse_area() -> "pd.Series":
    """meta_data.xlsx에서 온실 이름 → 면적(m²) 매핑을 반환한다(없으면 빈 Series)."""
    meta_path = get_data_dir() / "meta_data.xlsx"
    if not meta_path.exists():
        return pd.Series(dtype=float)
    meta = pd.read_excel(meta_path)
    if "name" not in meta.columns or "area (m2)" not in meta.columns:
        return pd.Series(dtype=float)
    # meta에 중복 name이 있어(예: PF_0023923_01) map() 시 InvalidIndexError가 나므로
    # 첫 행만 남겨 unique index를 보장한다.
    meta = meta.drop_duplicates(subset="name", keep="first")
    return meta.set_index("name")["area (m2)"].astype(float)


def area_tertile_labels(areas: "pd.Series") -> "pd.Series":
    """면적 Series를 3분위(Small/Medium/Large) 라벨 Series로 변환한다.

    온실 수가 적어(≈10) 분위 경계가 뭉치면 qcut이 실패할 수 있어, 실패 시
    median 기준 2분할로 폴백한다.
    """
    a = areas.dropna()
    if len(a) < 3:
        return pd.Series("All", index=a.index)
    try:
        return pd.qcut(a, 3, labels=["Small", "Medium", "Large"]).astype(str)
    except (ValueError, IndexError):
        med = a.median()
        return a.apply(lambda x: "Large" if x >= med else "Small")


def wavg_nmae(df: "pd.DataFrame", col: str = "NMAE") -> float:
    """n_eval 가중 평균(NaN 제외). 변수 단위 차이를 없앤 NMAE 집계에 사용."""
    m = df.dropna(subset=[col])
    if len(m) == 0 or m["n_eval"].sum() == 0:
        return float("nan")
    return float(np.average(m[col], weights=m["n_eval"]))


def agg_nmae(df: "pd.DataFrame", by, col: str = "NMAE") -> "pd.Series":
    """by 그룹별 n_eval 가중 NMAE."""
    return df.groupby(by, observed=True).apply(lambda g: wavg_nmae(g, col))


# ---------------------------------------------------------------------------
# 실험 설계 상수
# ---------------------------------------------------------------------------
LOSS_RATES = [0.1, 0.3, 0.5, 0.7, 0.9]

# 논문 v2 실험 설계
GAP_LENGTHS_H = [6, 12, 24, 72, 168]

GAP_LABELS = {
    6:   "6 h",
    12:  "12 h",
    24:  "24 h",
    72:  "72 h",
    168: "168 h",
}

SCENARIO_LABELS = {
    "A": "Scenario A\n(Individual)",
    "B": "Scenario B\n(Partial MV)",
    "C": "Scenario C\n(Full MV Block)",
}

CONTEXT_LENS = [24, 48, 96, 168, 336, 672]

CONTEXT_LABELS = {
    24:  "24h\n(1 day)",
    48:  "48h\n(2 days)",
    96:  "96h\n(4 days)",
    168: "168h\n(1 week)",
    336: "336h\n(2 weeks)",
    672: "672h\n(4 weeks)",
}


# ---------------------------------------------------------------------------
# 스타일 설정
# ---------------------------------------------------------------------------
def setup_figure_style() -> None:
    """모든 그림에 공통 적용되는 출판 품질 rcParams를 설정한다."""
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            # font.family를 "Arial"로 직접 지정하면 해당 폰트가 없는 시스템에서
            # font.sans-serif 대체 목록을 거치지 않고 곧장 DejaVu Sans로 폴백된다.
            # "sans-serif"(총칭 계열)로 지정해야 아래 목록이 실제로 적용된다.
            "font.family":      "sans-serif",
            "font.size":        11,
            "axes.labelsize":   12,
            "axes.titlesize":   13,
            "figure.dpi":       300,
            "savefig.dpi":      300,
            "savefig.bbox":     "tight",
            # 부수적이지만 출판 품질에 유리한 기본값
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "legend.frameon":    False,
            "pdf.fonttype":      42,   # 편집 가능한 폰트 (TrueType)
            "ps.fonttype":       42,
        }
    )
    # Arial이 설치되어 있지 않은 시스템(Linux 등)에서는 Arial과 자간·자폭이
    # 호환되도록 설계된 Liberation Sans로 대체한다(시각적으로 Arial과 거의 동일).
    plt.rcParams["font.sans-serif"] = [
        "Arial",
        "Liberation Sans",
        "DejaVu Sans",
        "sans-serif",
    ]
    # 음수 부호가 유니코드 마이너스 때문에 깨지지 않도록 처리
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 저장 헬퍼
# ---------------------------------------------------------------------------
def save_figure(fig, out_dir, filename: str) -> Path:
    """그림을 PDF 형식으로만 저장하고 경로를 출력한다.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    out_dir : str | Path
        저장 디렉토리 (없으면 생성).
    filename : str
        확장자 없는 파일명. 예: "Fig1_data_overview".

    Returns
    -------
    pdf_path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem  # 혹시 확장자가 붙어 와도 안전하게 제거
    pdf_path = out_dir / f"{stem}.pdf"

    fig.savefig(pdf_path)

    print(f"[saved] {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# 경로 헬퍼
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    """이 파일 기준 프로젝트 루트를 반환한다.

    구조: <root>/02_model/figures/fig_utils.py
    → parents[0]=figures, parents[1]=02_model, parents[2]=<root>
    """
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """원본 데이터 디렉토리 (01_data) 경로를 반환한다."""
    return _project_root() / "01_data"


def get_result_dir(subdir: str = "autogluon") -> Path:
    """결과 디렉토리 (03_result/<subdir>) 경로를 반환한다."""
    return _project_root() / "03_result" / subdir


def get_figure_dir() -> Path:
    """그림 출력 디렉토리 (04_figure) 경로를 반환한다."""
    return _project_root() / "04_figure"


# ---------------------------------------------------------------------------
# 모듈 import 시 곧바로 스타일 적용 (스크립트에서 잊더라도 안전)
# ---------------------------------------------------------------------------
setup_figure_style()

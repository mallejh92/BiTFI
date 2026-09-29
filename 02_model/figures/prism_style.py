"""Shared Prism-inspired publication style and immutable model colors."""
from pathlib import Path
import os,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[2]
STAGING=os.environ.get("BITFI_CLEAN")=="1"
CLEAN=STAGING or (os.environ.get("BITFI_CLEAN")!="0" and (ROOT/"03_result/active_evaluation.json").exists())
ACTIVE_RESULT=json.loads((ROOT/"03_result/active_evaluation.json").read_text()).get("result_root","03_result/reevaluation_clean_20260911") if (ROOT/"03_result/active_evaluation.json").exists() else "03_result/reevaluation_clean_20260911"
RESULT_ROOT=Path(os.environ["BITFI_RESULT_ROOT"]) if os.environ.get("BITFI_RESULT_ROOT") else (ROOT/ACTIVE_RESULT if CLEAN else ROOT/"03_result")
OUT=RESULT_ROOT/"figures" if STAGING else ROOT/"04_figure"
MAIN=["LI","SeasonalNaive","AG-LightGBM","SAITS","MOMENT-FT","TimesFM3.0","TimesFM3.0-COV-SPA","BiTFI-TimesFM3"]
COLORS={
 "LI":"#939AA3","SeasonalNaive":"#B49A77","AG-LightGBM":"#C59A35",
 "SAITS":"#8064AD","MOMENT-FT":"#146F69","MOMENT":"#289C96","TimesFM3.0":"#397DB8",
 "TimesFM3.0-COV-SPA":"#E18D4B","BiTFI-TimesFM3":"#C84E69",
 "BiTFI-Chronos2":"#A63353","BiTFI-TimesFM3-fwd":"#E2A0AE",
 "TimesFM3.0-COV":"#70A9CB","TimesFM3.0-MV":"#97BDD7",
 "TimesFM2.5":"#698D9A","Chronos2":"#4AABA8","Spatial-Ridge":"#858554",
 "CAFI":"#AC80A8","CAFI-R1":"#88608E","CAFI-TimesFM3":"#BA9CC8",
 "CAFI-TimesFM3-R1":"#9977AA","AG-RandomForest":"#9C873A",
 "AG-PatchTST":"#75819B","AG-DeepAR":"#B18A6D"}
LABELS={"LI":"Linear interpolation","SeasonalNaive":"Seasonal naive","AG-LightGBM":"LightGBM",
 "SAITS":"SAITS","MOMENT":"MOMENT (zero-shot)","MOMENT-FT":"MOMENT","TimesFM3.0":"TimesFM3 (univariate)",
 "TimesFM3.0-COV-SPA":"TimesFM3 (local + cross-greenhouse covariates)","BiTFI-TimesFM3":"BiTFI (TimesFM3)",
 "BiTFI-Chronos2":"BiTFI (Chronos 2)","BiTFI-TimesFM3-fwd":"BiTFI (TimesFM3, forward only)",
 "TimesFM3.0-COV":"TimesFM3 (local covariates)","TimesFM3.0-MV":"TimesFM3 (multivariate)",
 "TimesFM2.5":"TimesFM2.5 (univariate)","Chronos2":"Chronos 2 (multivariate)","Spatial-Ridge":"Spatial ridge",
 "CAFI":"CAFI (Chronos 2, R5)","CAFI-R1":"CAFI (Chronos 2, R1)",
 "CAFI-TimesFM3":"CAFI (TimesFM3, R5)","CAFI-TimesFM3-R1":"CAFI (TimesFM3, R1)",
 "AG-RandomForest":"Random forest","AG-PatchTST":"PatchTST","AG-DeepAR":"DeepAR"}
MARKERS=dict(zip(MAIN,["o","s","D","^","v","P","X","o"]))
VARS=["Tin","Tout","RH","CO2","Rad"]
VL={"Tin":"Indoor temperature","Tout":"Outdoor temperature","RH":"Relative humidity",
    "CO2":"CO₂ concentration","Rad":"Solar radiation"}
UNITS={"Tin":"°C","Tout":"°C","RH":"%","CO2":"ppm","Rad":r"W m$^{-2}$"}

# Identical comparison set and display names across main result figures.
import sys
sys.path.insert(0,str(ROOT/"02_model"))
from revision_config import MAIN_MODELS, LABELS as REVISION_LABELS, COLORS as REVISION_COLORS
MAIN=MAIN_MODELS
LABELS.update(REVISION_LABELS);COLORS.update(REVISION_COLORS)

def setup():
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],
      "font.size":8.5,"axes.titlesize":10,"axes.titleweight":"bold","axes.labelsize":9,
      "axes.linewidth":1.05,"axes.spines.top":False,"axes.spines.right":False,
      "xtick.direction":"out","ytick.direction":"out","xtick.major.size":3.5,"ytick.major.size":3.5,
      "xtick.major.width":1,"ytick.major.width":1,"xtick.labelsize":8,"ytick.labelsize":8,
      "axes.grid":False,"legend.frameon":False,"legend.fontsize":8,
      "lines.linewidth":1.6,"lines.markersize":4.5,"savefig.facecolor":"white",
      "figure.facecolor":"white","axes.facecolor":"white","pdf.fonttype":42,"ps.fonttype":42,
      "svg.fonttype":"none","axes.unicode_minus":True})

def panel(ax, letter, title=None):
    ax.text(-0.15,1.08,letter,transform=ax.transAxes,fontweight="bold",fontsize=13,va="top")
    if title:ax.set_title(title,loc="left",pad=10)

def tidy(ax):
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))

def interval(values, seed=42):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if len(x)<2:return (x.mean(),x.mean(),x.mean())
    rng=np.random.default_rng(seed)
    means=x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1)
    lo,hi=np.quantile(means,[.025,.975])
    return x.mean(),lo,hi

def finish_axes(fig):
    """Separate numerical x/y spines without changing limits or data artists."""
    for ax in fig.axes:
        if not ax.axison or ax.name != "rectilinear":
            continue
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.label.set_fontweight("semibold")
        ax.yaxis.label.set_fontweight("semibold")
        # Categorical dot plots already omit the y spine, as in Figure 5.
        if not (ax.spines["left"].get_visible() and ax.spines["bottom"].get_visible()):
            continue
        width = ax.get_position().width * fig.get_figwidth()
        offset = 3.0 if width < 1.2 else 5.0
        ax.spines["left"].set_position(("outward", offset))
        ax.spines["bottom"].set_position(("outward", offset))
        ax.tick_params(axis="both", which="both", direction="out", top=False, right=False)


def save(fig,name,supp=False):
    finish_axes(fig)
    folder=OUT/"Supplementary" if supp else OUT
    folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/f"{name}.pdf", bbox_inches="tight", pad_inches=.08)
    plt.close(fig)

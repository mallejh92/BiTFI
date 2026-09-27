"""Build publication figures and auditable source tables; no model training.

Main comparisons use the exact ctx1440 masks. Confidence intervals resample
greenhouses (n=10), not individual hours or overlapping evaluation strata.
Historical supplementary arms retain their originally reported protocols.
"""
from __future__ import annotations
import argparse, contextlib, io, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prism_style as ps
from Figure2_BiTFI_framework import main as build_framework
from final_table import SRC
from preprocessing import preprocess_file, TARGET_COLS

ROOT=ps.ROOT;OUT=ps.OUT;DATA=OUT/"source_data"
KEY=["greenhouse","scenario","masked_vars","gap_length_h","repeat","group_type","group_value","variable"]
NEWSRC={"MOMENT-FT":"comparison_moment-ft","SAITS":"comparison_saits","MOMENT":"comparison_moment","LI":"comparison_li",
        "SeasonalNaive":"comparison_seasonalnaive","AG-LightGBM":"comparison_ag-lightgbm"}

def load_results():
    sources={**SRC,**NEWSRC}
    if ps.CLEAN:
        sources={m:str(ps.RESULT_ROOT/"evaluation"/m) for m in sources}
        for m in ["SAITS-matched-local","SAITS-spatial"]:
            sources[m]=str(ROOT/"03_result/revision_experiments_20260926/saits_information/evaluation"/m)
    frames=[];cache={}
    for m,folder in sources.items():
        path=ROOT/"03_result"/folder/"results.csv"
        if not path.exists() and m in ["SAITS-matched-local","SAITS-spatial"]:
            path=ROOT/"03_result/revision_experiments_20260926/saits_information/evaluation"/(m+".csv")
        if not path.exists():raise FileNotFoundError(path)
        if folder not in cache:cache[folder]=pd.read_csv(path)
        d=cache[folder];d=d[d.model==m].copy()
        if d.empty:raise ValueError(f"Missing model {m}")
        if d.duplicated(KEY).any():raise ValueError(f"Duplicate cells: {m}")
        d["source_directory"]=folder;frames.append(d)
    full=pd.concat(frames,ignore_index=True)
    main=full[full.model.isin(ps.MAIN)].copy()
    ref=main[(main.model=="BiTFI-TimesFM3")&(main.group_type=="all")]
    refkeys=set(map(tuple,ref[KEY].to_numpy()))
    for m in ps.MAIN:
        d=main[(main.model==m)&(main.group_type=="all")]
        if set(map(tuple,d[KEY].to_numpy()))!=refkeys:
            raise ValueError(f"Incomplete or unmatched evaluation cells for {m}")
        if not np.isfinite(d.NMAE).all():raise ValueError(f"Non-finite scores {m}")
    return full,main

def site_scores(d,extra=()):
    keys=["model","greenhouse",*extra]
    z=d.assign(weighted=d.NMAE*d.n_eval).groupby(keys,observed=True)[["weighted","n_eval"]].sum()
    return (z.weighted/z.n_eval).rename("NMAE").reset_index()

def rowsummary(scores,models=ps.MAIN):
    rows=[]
    for m in models:
        x=scores[scores.model==m].NMAE.to_numpy()
        mu,lo,hi=ps.interval(x)
        rows.append(dict(model=m,label=ps.LABELS[m],NMAE=mu,SD=np.std(x,ddof=1),
                         CI_low=lo,CI_high=hi,n_greenhouses=len(x)))
    return pd.DataFrame(rows)

def dotplot(ax,scores,models=ps.MAIN,labels=True,points=True):
    rng=np.random.default_rng(14)
    for i,m in enumerate(models):
        x=scores[scores.model==m].sort_values("greenhouse").NMAE.to_numpy()
        mu,lo,hi=ps.interval(x);c=ps.COLORS[m]
        if points:
            jitter=rng.uniform(-.16,.16,len(x))
            ax.scatter(x,i+jitter,s=17,facecolors="white",edgecolors=c,lw=.9,alpha=.85,zorder=2)
        ax.errorbar(mu,i,xerr=[[mu-lo],[hi-mu]],fmt="D",ms=4.5,color=c,
                    elinewidth=1.65,capsize=3,capthick=1.2,zorder=3)
    ax.set_yticks(range(len(models)),[ps.LABELS[m].replace(" (", "\n(", 1) if labels else "" for m in models])
    ax.invert_yaxis();ax.set_xlabel("NMAE");ax.set_xlim(left=0)
    ax.tick_params(axis="y",length=0,pad=6,labelsize=7)
    ax.spines["left"].set_visible(False)

FIGURE3_MODELS=ps.MAIN

def overall(main):
    models=FIGURE3_MODELS
    a=main[(main.group_type=="all")&main.model.isin(models)]
    ref=set(map(tuple,a[a.model=="BiTFI-TimesFM3"][KEY].to_numpy()))
    for m in models:
        assert set(map(tuple,a[a.model==m][KEY].to_numpy()))==ref, m
    s=site_scores(a)
    s.to_csv(DATA/"main_greenhouse_scores.csv",index=False)
    tab=rowsummary(s,models);tab.to_csv(DATA/"main_summary.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.8),gridspec_kw={"width_ratios":[1.05,1]})
    fig.subplots_adjust(left=.32,right=.98,wspace=.48,bottom=.15,top=.86)
    dotplot(axes[0],s,models)
    axes[0].set_yticklabels([ps.LABELS[m].replace(" (","\n(",1) for m in models],fontsize=7.2)
    for i,m in enumerate(models):
        row=tab[tab.model==m].iloc[0]
        right=max(float(row.CI_high),float(s.loc[s.model==m,"NMAE"].max()))
        axes[0].annotate(f"{row.NMAE:.4f}",(right,i),xytext=(5,0),textcoords="offset points",
                         ha="left",va="center",fontsize=7,color="#303940",annotation_clip=False)
    axes[0].set_xlim(0,max(float(tab.CI_high.max()),float(s.NMAE.max()))*1.32)
    ps.panel(axes[0],"A","Reconstruction error")
    scores=s.pivot(index="greenhouse",columns="model",values="NMAE")
    bit=scores["BiTFI-TimesFM3"].to_numpy();records=[]
    for i,m in enumerate(models[:-1]):
        base=scores[m].to_numpy();c=ps.COLORS[m]
        rng=np.random.default_rng(42);ind=rng.integers(len(bit),size=(10000,len(bit)))
        boot=(1-bit[ind].mean(axis=1)/base[ind].mean(axis=1))*100
        value=(1-bit.mean()/base.mean())*100
        lo,hi=np.quantile(boot,[.025,.975])
        axes[1].errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt="o",color=c,capsize=3,ms=4.5)
        axes[1].annotate(f"{value:.1f}%",(hi,i),xytext=(5,0),textcoords="offset points",
                         ha="left",va="center",fontsize=7,color="#303940",annotation_clip=False)
        p=float(wilcoxon(base,bit).pvalue)
        records.append(dict(reference=m,reduction_percent=value,CI_low=lo,CI_high=hi,
                            wilcoxon_p=p,n_greenhouses=len(bit)))
    tests=pd.DataFrame(records)
    order=np.argsort(tests.wilcoxon_p.to_numpy());adjusted=np.zeros(len(tests));last=0
    for rank,idx in enumerate(order):
        last=max(last,(len(tests)-rank)*tests.loc[idx,"wilcoxon_p"]);adjusted[idx]=min(1,last)
    tests["holm_p"]=adjusted;tests.to_csv(DATA/"paired_greenhouse_tests.csv",index=False)
    axes[1].set_yticks(range(len(models)-1),["" for _ in models[:-1]])
    axes[0].set_ylim(len(models)-.5,-.5);axes[1].set_ylim(len(models)-.5,-.5);axes[1].tick_params(axis="y",length=0);axes[1].spines["left"].set_visible(False)
    axes[1].set_xlim(min(0,float(tests.CI_low.min()))-3,float(tests.CI_high.max())*1.28)
    axes[1].axvline(0,color="#BBBBBB",lw=.8,zorder=0)
    axes[1].set_xlabel("Error reduction with BiTFI (%)")
    ps.panel(axes[1],"B","BiTFI improvement")
    ps.save(fig,"Figure3_Overall_performance")

FIGURE45_MODELS=ps.MAIN
FIGURE456_LABELS=ps.LABELS | {
    "SAITS-spatial": "SAITS", "MOMENT-FT": "MOMENT",
    "TimesFM3.0-COV-SPA": "TimesFM3", "BiTFI-TimesFM3": "BiTFI",
}

def figure45_results(full):
    selected=full[full.model.isin(FIGURE45_MODELS)].copy()
    ref=selected[selected.model=="BiTFI-TimesFM3"].set_index(KEY).n_eval.sort_index()
    for model in FIGURE45_MODELS:
        got=selected[selected.model==model].set_index(KEY).n_eval.sort_index()
        pd.testing.assert_series_equal(got,ref,check_names=False)
    return selected

def gap_robustness(full):
    main=figure45_results(full)
    a=site_scores(main[main.group_type=="all"],["scenario","gap_length_h"])
    a.to_csv(DATA/"gap_scenario_greenhouse_scores.csv",index=False)
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.8),sharey=True)
    fig.subplots_adjust(left=.09,right=.98,wspace=.24,top=.80,bottom=.17)
    for j,sc in enumerate(["A","B","C"]):
        ax=axes[j]
        for m in FIGURE45_MODELS:
            z=a[(a.model==m)&(a.scenario==sc)].groupby("gap_length_h").NMAE.mean().reindex([6,12,24,72,168])
            ax.plot(np.arange(5),z,color=ps.COLORS[m],marker=ps.MARKERS.get(m,"h"),
                    lw=2 if m=="BiTFI-TimesFM3" else 1.2,ms=4,label=FIGURE456_LABELS[m])
        ax.set_xticks(range(5),["6","12","24","72","168"]);ax.set_xlabel("Gap length (h)")
        ax.set_ylim(bottom=0)
        ps.panel(ax,chr(65+j),{"A":"Single sensor","B":"Indoor sensor group","C":"All sensors"}[sc])
    axes[0].set_ylabel("NMAE")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,[x.replace(" (","\n(",1).replace("local + cross-greenhouse covariates","local + cross-greenhouse\ncovariates") for x in labels],ncol=4,loc="upper center",bbox_to_anchor=(.53,1.02),columnspacing=1.2,handlelength=1.7,fontsize=7.5)
    ps.save(fig,"Figure4_Gap_robustness")

def variables(full):
    main=figure45_results(full)
    s=site_scores(main[main.group_type=="all"],["variable"])
    s.to_csv(DATA/"variable_greenhouse_scores.csv",index=False)
    fig,axes=plt.subplots(2,3,figsize=(7.2,6.3))
    fig.subplots_adjust(left=.25,right=.98,wspace=.65,hspace=.6,top=.92,bottom=.10)
    for i,v in enumerate(ps.VARS):
        ax=axes.flat[i];dotplot(ax,s[s.variable==v],models=FIGURE45_MODELS,labels=False,points=False)
        if i%3==0:ax.set_yticklabels([FIGURE456_LABELS[m].replace(" (","\n(",1) for m in FIGURE45_MODELS])
        ps.panel(ax,chr(65+i),ps.VL[v])
    axes.flat[5].axis("off")
    ps.save(fig,"Figure5_Variable_performance")

def overview():
    cache=DATA/"dataset_overview.csv"
    if cache.exists():d=pd.read_csv(cache,parse_dates=["start","end"])
    else:
        split=json.loads((ROOT/"03_result/comparison/split.json").read_text());rows=[]
        for group in ["train","test"]:
            for fp in split[group]:
                with contextlib.redirect_stdout(io.StringIO()):r=preprocess_file(fp)
                if r is None:continue
                a=r["data"][ps.VARS]
                rows.append(dict(greenhouse=r["name"],group=group,start=a.index.min(),end=a.index.max(),
                                 n_rows=len(a),**{v:100*a[v].isna().mean() for v in ps.VARS}))
        d=pd.DataFrame(rows).sort_values(["group","start"],ascending=[False,True])
        d["display_id"]=[("Train" if g=="train" else "Test")+f" {i+1:02d}" for g,z in d.groupby("group",sort=False) for i in range(len(z))]
        d.to_csv(cache,index=False)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,7.0),gridspec_kw={"width_ratios":[2.2,1]})
    fig.subplots_adjust(left=.12,right=.90,wspace=.27,top=.9,bottom=.1)
    for i,r in enumerate(d.itertuples()):
        c="#AEB9C5" if r.group=="train" else "#397DB8"
        ax.barh(i,(r.end-r.start).total_seconds()/86400,left=mdates.date2num(r.start),height=.62,color=c)
    ax.set_yticks(range(len(d)),d.display_id,fontsize=7);ax.invert_yaxis()
    ax.set_ylim(len(d)-.5,-.5)
    ax.tick_params(axis="y",length=0);ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlabel("Recorded coverage")
    im=bx.imshow(d[ps.VARS].to_numpy(),aspect="auto",cmap="Greys",vmin=0,vmax=100,interpolation="nearest")
    bx.set_xticks(range(5),["Tin","Tout","RH","CO₂","Rad"]);bx.set_yticks([])
    for sp in bx.spines.values():sp.set_visible(False)
    cax=fig.add_axes([.925,.28,.017,.40]);cb=fig.colorbar(im,cax=cax);cb.set_label("Missing after preprocessing (%)",fontsize=8)
    ps.panel(ax,"A","Cross-greenhouse evaluation");ps.panel(bx,"B","Sensor availability")
    ps.save(fig,"Figure1_Dataset_overview")

FIGURE6_STYLE = {
    "observed": dict(color="#BDBDBD", lw=.9, alpha=1., zorder=1),
    "gap": dict(color="#E8F1F8", alpha=1., zorder=0),
    "boundary": dict(color="#7FA6C9", lw=.8, ls=":", zorder=1),
    "Spatial-Ridge": dict(color=ps.COLORS["Spatial-Ridge"], lw=1., alpha=.55, ls="-.", marker=None, zorder=2),
    "MOMENT-FT": dict(color="#7FB8AE", lw=1., alpha=.55, ls="-", marker=None, zorder=2),
    "TimesFM3.0-COV-SPA": dict(color=ps.COLORS["TimesFM3.0-COV-SPA"], lw=1.4, alpha=.9, ls="-", zorder=3),
    "SAITS-spatial": dict(color="#A99BC9", lw=1., alpha=.55, ls="--", marker=None, zorder=2),
    "BiTFI-TimesFM3": dict(color="#D6336C", lw=2., alpha=1., ls="-", marker="o",
                         ms=3.5, markevery=12, markeredgecolor="white", markeredgewidth=.4, zorder=4),
    "truth": dict(color="#111111", ls=(0,(3,2)), lw=1.3, alpha=.9, zorder=5),
    "tick": dict(width=.8, length=3.5, direction="out", colors="#222222", labelsize=7, pad=2),
    "spine": dict(linewidth=.8, color="#222222"),
    "annotation": dict(fontsize=6.5, color="#222222",
                       bbox=dict(facecolor="white", edgecolor="#999999", linewidth=.5, alpha=.9, pad=1.5)),
    "rc": {"font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","Liberation Sans"],
           "axes.labelsize":8, "pdf.fonttype":42},
}

def _figure6_score_box(ax, bit, base, baseline_name):
    """Compact three-column table with ordinary, unscaled font glyphs."""
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, VPacker, TextArea
    columns=[("", "BiTFI", "Best"),
             ("R²", f"{bit['R2']:.2f}", f"{base['R2']:.2f}"),
             ("MAE", f"{bit['MAE']:.2f}", f"{base['MAE']:.2f}")]
    packed=[]
    for i,column in enumerate(columns):
        cells=[TextArea(value,textprops=dict(fontfamily="Liberation Sans",fontsize=7.5,
                    color="#222222",fontweight="bold" if row==0 else "normal"))
               for row,value in enumerate(column)]
        packed.append(VPacker(children=cells,align="left" if i==0 else "right",pad=0,sep=1.5))
    header=TextArea("Best: "+baseline_name,textprops=dict(fontfamily="Liberation Sans",fontsize=7.5,color="#222222"))
    content=VPacker(children=[header,HPacker(children=packed,align="top",pad=0,sep=4)],align="left",pad=0,sep=2)
    box=AnchoredOffsetbox(loc="upper right",child=content,
                         bbox_to_anchor=(1.,.98),bbox_transform=ax.transAxes,
                         pad=.2,borderpad=.2,frameon=True,prop=dict(size=6.5))
    box.patch.set(facecolor="white",edgecolor="#999999",linewidth=.5,alpha=.9)
    box.set_zorder(10);ax.add_artist(box)
    return box

@plt.rc_context(FIGURE6_STYLE["rc"])
def examples():
    """Legacy-style scenario rows × sensor columns, using one fixed 72 h gap."""
    st=FIGURE6_STYLE; metrics=[]; annotations=[]
    folder=ps.RESULT_ROOT/"figure6_common_window"
    names=["SAITS-spatial","MOMENT-FT","Spatial-Ridge","TimesFM3.0-COV-SPA","BiTFI"]
    frames=[pd.read_csv(folder/f"{m}.csv",parse_dates=["datetime"]) for m in names]
    data=pd.concat(frames,ignore_index=True)
    data.to_csv(DATA/"illustrative_case_ABC_physical_units.csv",index=False)
    data[data.scenario=="C"].to_csv(DATA/"illustrative_case_physical_units.csv",index=False)
    assert not data.duplicated(["model","scenario","variable","datetime"]).any()
    models=["SAITS-spatial","MOMENT-FT","Spatial-Ridge","TimesFM3.0-COV-SPA","BiTFI-TimesFM3"]
    fig=plt.figure(figsize=(7.2,7.0))
    grid=fig.add_gridspec(6,5,height_ratios=[.72,1.]*3,
                          left=.085,right=.99,bottom=.09,top=.865,wspace=.31,hspace=.16)
    axes=np.empty((3,5),dtype=object);headers=np.empty((3,5),dtype=object)
    for ri in range(3):
        for ci in range(5):
            headers[ri,ci]=fig.add_subplot(grid[2*ri,ci]);headers[ri,ci].set_axis_off()
            axes[ri,ci]=fig.add_subplot(grid[2*ri+1,ci])
    titles=[r"$T_{\mathrm{in}}$ (°C)",r"$T_{\mathrm{out}}$ (°C)","RH (%)",r"CO$_2$ (ppm)",r"Rad (W m$^{-2}$)"]
    for ri,sc in enumerate(["A","B","C"]):
        for ci,v in enumerate(ps.VARS):
            ax=axes[ri,ci]
            z=data.query("model=='BiTFI-TimesFM3' and scenario==@sc and variable==@v").sort_values("hours")
            x=z.hours.to_numpy();truth=z.truth.to_numpy();gap=z.artificial.to_numpy(bool)
            if gap.any():
                ax.axvspan(0,72,**st["gap"])
                for boundary in [0,72]:
                    ax.axvline(boundary,**st["boundary"])
                ax.plot(x,np.where(gap,np.nan,truth),**st["observed"])
                ax.plot(x[gap],truth[gap],**st["truth"])
                for m in models:
                    q=data.query("model==@m and scenario==@sc and variable==@v").sort_values("hours")
                    np.testing.assert_array_equal(q.hours,x)
                    np.testing.assert_allclose(q.truth,truth,equal_nan=True)
                    np.testing.assert_array_equal(q.artificial,gap)
                    prediction=q.prediction.to_numpy()[gap]
                    ax.plot(x[gap],prediction,**st[m])
                    y=truth[gap]
                    if not (np.isfinite(y).all() and np.isfinite(prediction).all()):
                        raise ValueError("Nonfinite gap values")
                    residual=np.sum((y-prediction)**2);total=np.sum((y-y.mean())**2)
                    r2=1-residual/total if total>0 else (1. if residual==0 else 0.)
                    metrics.append(dict(panel=chr(65+ri*5+ci),scenario=sc,variable=v,model=m,
                                        R2=r2,MAE=np.mean(np.abs(y-prediction))))
                scores={m:next(r for r in reversed(metrics) if r["scenario"]==sc and r["variable"]==v and r["model"]==m) for m in models}
                best=min(["MOMENT-FT","SAITS-spatial","Spatial-Ridge","TimesFM3.0-COV-SPA"],key=lambda m:scores[m]["MAE"])
                bit=scores["BiTFI-TimesFM3"];base=scores[best]
                annotations.append(_figure6_score_box(headers[ri,ci],bit,base,FIGURE456_LABELS[best]))

            else:
                ax.set_facecolor("#F7F8F9")
                ax.plot(x,truth,color="#59636C",alpha=.22,lw=.75,zorder=2)
                ax.text(.5,.5,"Observed covariate\n(not masked)",ha="center",va="center",
                        transform=ax.transAxes,fontsize=6.5,color="#68747D",
                        bbox=dict(facecolor="#F7F8F9",edgecolor="none",alpha=.92,pad=2))
            headers[ri,ci].text(-.06,.86,chr(65+ri*5+ci),transform=headers[ri,ci].transAxes,
                    fontweight="bold",fontsize=9,va="bottom")
            ax.set_xlim(-72,96);ax.set_xticks([-72,0,72])
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.minorticks_off()
            ax.tick_params(axis="both",which="major",**st["tick"])
            ax.tick_params(top=False,right=False)
            for spine in ax.spines.values():spine.set(**st["spine"])
            ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
            if ri==0:headers[ri,ci].set_title(titles[ci],fontsize=8.5,pad=7,weight="normal")
            if ri<2:ax.tick_params(labelbottom=False)
            if ci==0:ax.set_ylabel(f"Scenario {sc}",fontsize=8,labelpad=7)
    handles=[Line2D([0],[0],label="Observed context",**st["observed"]),
             __import__("matplotlib").patches.Patch(label="72 h gap",**st["gap"]),
             Line2D([0],[0],label="Withheld truth",**st["truth"])] + [
             Line2D([0],[0],label=("BiTFI" if m=="BiTFI-TimesFM3" else FIGURE456_LABELS[m]),**st[m]) for m in ["Spatial-Ridge","MOMENT-FT","SAITS-spatial","TimesFM3.0-COV-SPA","BiTFI-TimesFM3"]]
    fig.legend(handles=handles,ncol=4,loc="upper center",bbox_to_anchor=(.53,.998),
               fontsize=7,columnspacing=.8,handlelength=2.5,handletextpad=.4)
    fig.text(.53,.025,"Hours from gap start",ha="center",fontsize=8)
    # Include every displayed prediction; never retain limits from another case.
    for ci,v in enumerate(ps.VARS):
        q=data[data.variable==v];values=np.r_[q.truth.to_numpy(),q.prediction.to_numpy()];values=values[np.isfinite(values)]
        lo,hi=float(values.min()),float(values.max());pad=max((hi-lo)*.07,1e-3)
        for ax in axes[:,ci]:ax.set_xlim(-72,96);ax.set_ylim(lo-pad,hi+pad)
    fig.canvas.draw()
    for item in annotations:
        bbox=item.get_window_extent(fig.canvas.get_renderer());ab=item.axes.get_window_extent()
        if bbox.x0 < ab.x0 or bbox.x1 > ab.x1 or bbox.y0 < ab.y0 or bbox.y1 > ab.y1:
            raise ValueError(f"Score box exceeds its dedicated header: {bbox.bounds} vs {ab.bounds}")
        if any(bbox.overlaps(axis.get_window_extent()) for axis in axes.flat):
            raise ValueError("Score box overlaps a time-series plotting area")
    (DATA/"figure6_annotation_layout.json").write_text(json.dumps(dict(
        score_boxes=len(annotations),dedicated_header_axes=True,overlap_with_plot_axes=False,
        masked_panels=13,display_context_hours_before=72,display_context_hours_after=24),indent=2))
    fig.savefig(OUT/"Figure6_Reconstruction_example.pdf",bbox_inches="tight",pad_inches=.08)
    table=pd.DataFrame(metrics)
    table.to_csv(DATA/"figure6_panel_metrics.csv",index=False)
    print(table.to_csv(index=False))
    limits=[dict(xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim())) for ax in axes.flat]
    plt.close(fig)
    return table,limits

def supplementary(full,main):
    s=site_scores(full[full.group_type=="all"])
    extended=s[~s.model.str.startswith("CAFI")]
    order=extended.groupby("model").NMAE.mean().sort_values().index.tolist()
    rowsummary(extended,order).to_csv(DATA/"all_model_summary.csv",index=False)
    fig,ax=plt.subplots(figsize=(7.2,8.0));fig.subplots_adjust(left=.45,bottom=.12,top=.94,right=.96)
    dotplot(ax,extended,order)
    ax.set_title("Extended model comparison",loc="left",pad=15)
    ps.save(fig,"FigureS1_Extended_comparison",supp=True)
    if (ps.RESULT_ROOT/"protocol.json").exists() and "selected_contexts" in json.loads((ps.RESULT_ROOT/"protocol.json").read_text()):
        from FigureS2_validation_context import main as build_context_comparison
    else:
        from FigureS2_controlled_context import main as build_context_comparison
    build_context_comparison()
    h=pd.read_csv((ps.RESULT_ROOT/"models/SAITS/history.csv" if ps.CLEAN else ROOT/"03_result/comparison_saits/models/SAITS/history.csv"));best=h.loc[h.validation_mae.idxmin()]
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1));fig.subplots_adjust(left=.10,right=.97,bottom=.21,top=.84,wspace=.32)
    for ax,col,title,letter in zip(axes,["train_loss","validation_mae"],["Training objective","Held-out validation error"],["A","B"]):
        ax.plot(h.epoch,h[col],color=ps.COLORS["SAITS"]);ax.axvline(best.epoch,color="#888888",ls="--",lw=.8)
        ax.set_xlabel("Epoch");ax.set_ylabel("ORT + MIT loss" if col=="train_loss" else "Masked MAE (normalized scale)")
        ps.panel(ax,letter,title)
    axes[1].scatter([best.epoch],[best.validation_mae],color=ps.COLORS["SAITS"],s=25,zorder=3)
    ps.save(fig,"FigureS3_SAITS_training",supp=True)
    seas=site_scores(main[main.group_type=="season"],["group_value"])
    seas.to_csv(DATA/"season_greenhouse_scores.csv",index=False)
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.2));fig.subplots_adjust(left=.32,right=.98,wspace=.27,hspace=.45,bottom=.10,top=.92)
    for i,season in enumerate(["spring","summer","fall","winter"]):
        ax=axes.flat[i];dotplot(ax,seas[seas.group_value==season],labels=i%2==0,points=False)
        if i%2==0:ax.set_yticklabels([ps.LABELS[m].replace(" (","\n(",1) for m in ps.MAIN],fontsize=7)
        ps.panel(ax,chr(65+i),season.capitalize())
    ps.save(fig,"FigureS4_Seasonal_performance",supp=True)
    models=["TimesFM3.0","TimesFM3.0-COV","TimesFM3.0-COV-SPA","BiTFI-TimesFM3"]
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5));fig.subplots_adjust(left=.28,right=.98,wspace=.65,bottom=.20,top=.84)
    dotplot(axes[0],s,models,points=True);ps.panel(axes[0],"A","Available information")
    v=site_scores(full[(full.group_type=="all")&full.model.isin(["Spatial-Ridge","TimesFM3.0-COV-SPA","BiTFI-TimesFM3"])],["variable"])
    for m in ["Spatial-Ridge","TimesFM3.0-COV-SPA","BiTFI-TimesFM3"]:
        z=v[v.model==m].copy();z["type"]=np.where(z.variable.isin(["Tin","RH","CO2"]),"Indoor","Outdoor")
        q=z.groupby(["greenhouse","type"]).NMAE.mean().reset_index()
        means=[q[q.type==t].NMAE.mean() for t in ["Indoor","Outdoor"]]
        axes[1].plot([0,1],means,marker="o",color=ps.COLORS[m],label=ps.LABELS[m])
    axes[1].set_xticks([0,1],["Indoor","Outdoor"]);axes[1].set_ylabel("NMAE")
    handles,labels=axes[1].get_legend_handles_labels()
    axes[1].legend(handles,[x.replace(" (","\n(",1).replace("local + cross-greenhouse covariates","local + cross-greenhouse\ncovariates") for x in labels],fontsize=6.3,loc="upper right");ps.panel(axes[1],"B","Cross-greenhouse regression")
    ps.save(fig,"FigureS5_Information_sources",supp=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--overview-only",action="store_true");args=p.parse_args()
    if ps.CLEAN and not args.overview_only:
        if (ps.RESULT_ROOT/"protocol.json").exists() and "selected_contexts" in json.loads((ps.RESULT_ROOT/"protocol.json").read_text()):
            import runpy
            runpy.run_path(str(ROOT/"02_model/render_selected_context_figures.py"), run_name="__main__")
            return
        from finalize_clean_figures import main as finalize
        finalize();return
    DATA.mkdir(parents=True,exist_ok=True);ps.setup()
    overview()
    if args.overview_only:return
    full,main=load_results()
    full.groupby("model").agg(rows=("NMAE","size"),source=("source_directory","first"),
        context_len=("context_len","first")).to_csv(DATA/"result_sources.csv")
    (OUT/"model_palette.json").write_text(json.dumps({m:{"label":ps.LABELS[m],"hex":ps.COLORS[m]} for m in ps.COLORS},indent=2))
    build_framework();overall(full);gap_robustness(full);variables(full);examples();supplementary(full,main)
    from FigureS7_univariate_backbones import main as build_backbone_comparison
    build_backbone_comparison()
    from FigureS8_moment_tuning import main as build_moment_tuning
    build_moment_tuning()
    print("Saved 6 main + 7 supplementary figures (PDF only), source tables and palette.",flush=True)

if __name__=="__main__":main()

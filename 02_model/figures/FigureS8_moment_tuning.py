"""PDF-only MOMENT head tuning diagnostics."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
sys.path.insert(0,str(Path(__file__).resolve().parent))
import prism_style as ps
from build_prism_figures import site_scores
ROOT=ps.ROOT;OUT=ps.OUT
def main():
    ps.setup();training=ps.RESULT_ROOT/"models/MOMENT" if ps.CLEAN else ROOT/"03_result/comparison_moment-ft/models/MOMENT"
    history=pd.read_csv(training/"history.csv");search=pd.read_csv(training/"search.csv")
    frames=[]
    for m,f in [("MOMENT","comparison_moment"),("MOMENT-FT","comparison_moment-ft")]:
        z=pd.read_csv(ps.RESULT_ROOT/"evaluation"/m/"results.csv" if ps.CLEAN else ROOT/"03_result"/f/"results.csv");frames.append(z[(z.model==m)&(z.group_type=="all")])
    full=pd.concat(frames);scores=site_scores(full);data=OUT/"source_data"
    scores.to_csv(data/"moment_tuning_greenhouse_scores.csv",index=False)
    pivot=scores.pivot(index="greenhouse",columns="model",values="NMAE");a=pivot.MOMENT.to_numpy();b=pivot["MOMENT-FT"].to_numpy()
    rng=np.random.default_rng(42);ix=rng.integers(len(a),size=(10000,len(a)));boot=100*(1-b[ix].mean(axis=1)/a[ix].mean(axis=1));lo,hi=np.quantile(boot,[.025,.975])
    stats=dict(zero_shot_NMAE=a.mean(),head_tuned_NMAE=b.mean(),relative_reduction_percent=100*(1-b.mean()/a.mean()),CI_low=lo,CI_high=hi,paired_wilcoxon_p=float(wilcoxon(a,b).pvalue),improved_greenhouses=int((b<a).sum()),n_greenhouses=len(a))
    (data/"moment_tuning_statistics.json").write_text(json.dumps(stats,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.8));fig.subplots_adjust(left=.11,right=.97,bottom=.11,top=.9,hspace=.6,wspace=.35)
    ax=axes[0,0]
    for lr,c in zip(sorted(history.learning_rate.unique()),["#AECFC9","#5AA69B","#146F69","#71859D","#C79252"]):
        d=history[history.learning_rate==lr];ax.plot(d.epoch,d.validation_mae,color=c,lw=1.5,label=f"lr = {lr:g}")
    zero=history.zero_shot_validation_mae.iloc[0];ax.axhline(zero,color="#888888",ls="--",lw=.8,label="Zero-shot")
    best=search.loc[search.best_validation_mae.idxmin()];ax.scatter([best.best_epoch],[best.best_validation_mae],marker="*",s=60,c="#146F69",zorder=5)
    ax.set_xlabel("Epoch");ax.set_ylabel("Validation masked MAE");ax.legend(fontsize=6.8);ps.panel(ax,"A","Validation selection")
    ax=axes[0,1]
    for aa,bb in zip(a,b):ax.plot([0,1],[aa,bb],color="#D5DADF",lw=.8,zorder=1)
    for i,(m,x) in enumerate([("MOMENT",a),("MOMENT-FT",b)]):
        mu,lo,hi=ps.interval(x);c=ps.COLORS[m]
        ax.scatter(np.full(len(x),i),x,facecolors="white",edgecolors=c,s=18,zorder=2)
        ax.errorbar(i,mu,yerr=[[mu-lo],[hi-mu]],fmt="D",color=c,capsize=3,ms=5,zorder=3)
    ax.set_xticks([0,1],["Zero-shot","Head-tuned"]);ax.set_xlim(-.4,1.4);ax.set_ylim(bottom=0);ax.set_ylabel("Test NMAE");ps.panel(ax,"B","Held-out test accuracy")
    for ax,col,levels,labels,letter,title in [
        (axes[1,0],"gap_length_h",[6,12,24,72,168],["6","12","24","72","168"],"C","Gap duration"),
        (axes[1,1],"variable",ps.VARS,[r"$T_\mathrm{in}$",r"$T_\mathrm{out}$","RH",r"CO$_2$","Rad"],"D","Sensor variables")]:
        z=site_scores(full,[col]);z.to_csv(data/f"moment_tuning_{col}.csv",index=False)
        for i,m in enumerate(["MOMENT","MOMENT-FT"]):
            mus=[];low=[];high=[]
            for v in levels:
                mu,lo,hi=ps.interval(z[(z.model==m)&(z[col]==v)].NMAE);mus.append(mu);low.append(mu-lo);high.append(hi-mu)
            ax.errorbar(np.arange(len(levels))+(i-.5)*.08,mus,yerr=[low,high],color=ps.COLORS[m],marker=["s","o"][i],ms=3.5,lw=1.4,capsize=2,label=["Zero-shot","Head-tuned"][i])
        ax.set_xticks(range(len(levels)),labels);ax.set_ylim(bottom=0);ax.set_ylabel("Test NMAE");ax.set_xlabel("Gap length (h)" if col=="gap_length_h" else "Variable")
        ax.legend(fontsize=7);ps.panel(ax,letter,title)
    ps.save(fig,"FigureS7_MOMENT_head_tuning",supp=True);print(json.dumps(stats,indent=2))
if __name__=="__main__":main()

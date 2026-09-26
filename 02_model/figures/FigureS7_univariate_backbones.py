"""PDF-only controlled univariate backbone comparison, Figure S6."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
sys.path.insert(0,str(Path(__file__).resolve().parent))
import prism_style as ps
from build_prism_figures import site_scores
ROOT=ps.ROOT;OUT=ps.OUT;SOURCE=ps.RESULT_ROOT/"univariate_backbone_comparison"
MODELS=["Chronos2-UNI","TimesFM2.5-UNI","TimesFM3.0-UNI"]
LABELS=["Chronos 2","TimesFM2.5","TimesFM3"]
COLORS=[ps.COLORS["Chronos2"],ps.COLORS["TimesFM2.5"],ps.COLORS["TimesFM3.0"]]
def main():
    ps.setup()
    full=pd.concat([pd.read_csv(SOURCE/f"{m[:-4]}_results.csv") for m in MODELS],ignore_index=True)
    key=["greenhouse","scenario","masked_vars","gap_length_h","repeat","variable"]
    ref=set(map(tuple,full[full.model==MODELS[0]][key].to_numpy()))
    for m in MODELS:
        d=full[full.model==m];assert set(map(tuple,d[key].to_numpy()))==ref and not d.duplicated(key).any()
        assert len(d)==6085 and np.isfinite(d.NMAE).all()
    sites=site_scores(full);summ=[]
    for m,label in zip(MODELS,LABELS):
        vals=sites.query("model==@m").NMAE.to_numpy();mean,lo,hi=ps.interval(vals)
        summ.append(dict(model=m,label=label,NMAE=mean,CI_low=lo,CI_high=hi,SD=vals.std(ddof=1),n_greenhouses=len(vals)))
    summary=pd.DataFrame(summ)
    folder=OUT/"source_data";summary.to_csv(folder/"univariate_backbone_summary.csv",index=False);sites.to_csv(folder/"univariate_backbone_greenhouse_scores.csv",index=False)
    pivot=sites.pivot(index="greenhouse",columns="model",values="NMAE");tests=[]
    for m in MODELS[:-1]:
        a=pivot[m].to_numpy();b=pivot[MODELS[-1]].to_numpy()
        rng=np.random.default_rng(42);ix=rng.integers(0,len(a),size=(10000,len(a)))
        boot=100*(1-b[ix].mean(axis=1)/a[ix].mean(axis=1));lo,hi=np.quantile(boot,[.025,.975])
        tests.append(dict(reference=m,TFM3_reduction_percent=100*(1-b.mean()/a.mean()),CI_low=lo,CI_high=hi,p=float(wilcoxon(a,b).pvalue)))
    tests=pd.DataFrame(tests);order=np.argsort(tests.p);adj=np.empty(2);last=0
    for rank,idx in enumerate(order):last=max(last,(2-rank)*tests.loc[idx,"p"]);adj[idx]=min(1,last)
    tests["holm_p"]=adj;tests.to_csv(folder/"univariate_backbone_paired_tests.csv",index=False)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.8))
    fig.subplots_adjust(left=.17,right=.97,bottom=.12,top=.86,wspace=.43,hspace=.65)
    ax=axes[0,0];rng=np.random.default_rng(42)
    for i,m in enumerate(MODELS):
        vals=sites[sites.model==m].NMAE.to_numpy();row=summary.iloc[i]
        ax.scatter(vals,i+rng.uniform(-.12,.12,len(vals)),s=18,facecolors="white",edgecolors=COLORS[i],lw=.8)
        ax.errorbar(row.NMAE,i,xerr=[[row.NMAE-row.CI_low],[row.CI_high-row.NMAE]],fmt="D",color=COLORS[i],ms=4,capsize=3,lw=1.5)
    ax.set_yticks(range(3),LABELS,fontsize=8);ax.set_ylim(2.6,-.6);ax.set_xlim(left=0);ax.set_xlabel("NMAE")
    ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0);ps.panel(ax,"A","Univariate accuracy")
    for ax,col,levels,labels,title,letter in [
        (axes[0,1],"gap_length_h",[6,12,24,72,168],["6","12","24","72","168"],"Gap duration","B"),
        (axes[1,0],"variable",ps.VARS,[r"$T_\mathrm{in}$",r"$T_\mathrm{out}$","RH",r"CO$_2$","Rad"],"Sensor variables","C"),
        (axes[1,1],"scenario",["A","B","C"],["A","B","C"],"Failure scenarios","D")]:
        scores=site_scores(full,[col]);scores.to_csv(folder/f"univariate_backbone_{col}.csv",index=False)
        for i,m in enumerate(MODELS):
            means=[];los=[];his=[]
            for level in levels:
                vals=scores[(scores.model==m)&(scores[col]==level)].NMAE.to_numpy()
                mu,lo,hi=ps.interval(vals);means.append(mu);los.append(mu-lo);his.append(hi-mu)
            xs=np.arange(len(levels))+(i-1)*.07
            ax.errorbar(xs,means,yerr=[los,his],color=COLORS[i],marker=["s","^","o"][i],lw=1.3,ms=3.5,capsize=2,label=LABELS[i])
        ax.set_xticks(range(len(levels)),labels);ax.set_ylabel("NMAE");ax.set_ylim(bottom=0)
        ax.set_xlabel({"gap_length_h":"Gap length (h)","variable":"Variable","scenario":"Scenario"}[col]);ps.panel(ax,letter,title)
    # Separate framework-level check from the controlled univariate experiment.
    ax=axes[1,1];ax.clear();frames=[]
    for model,folder_name in [("BiTFI-Chronos2","comparison_bitfi_chronos"),("BiTFI-TimesFM3","comparison_bitfi_tfm3")]:
        z=pd.read_csv(ps.RESULT_ROOT/"evaluation"/model/"results.csv" if ps.CLEAN else ROOT/"03_result"/folder_name/"results.csv")
        z=z[(z.model==model)&(z.group_type=="all")];frames.append(z)
    assert set(map(tuple,frames[0][key].to_numpy()))==set(map(tuple,frames[1][key].to_numpy()))
    framework=site_scores(pd.concat(frames));framework.to_csv(folder/"bitfi_backbone_greenhouse_scores.csv",index=False)
    paired=framework.pivot(index="greenhouse",columns="model",values="NMAE")
    av=paired["BiTFI-Chronos2"].to_numpy();bv=paired["BiTFI-TimesFM3"].to_numpy()
    for aa,bb in zip(av,bv):ax.plot([0,1],[aa,bb],color="#D3D7DC",lw=.7,zorder=1)
    for i,vals in enumerate([av,bv]):
        mu,lo,hi=ps.interval(vals);c=COLORS[0 if i==0 else 2]
        ax.scatter(np.full(len(vals),i),vals,s=17,facecolors="white",edgecolors=c,lw=.8,zorder=2)
        ax.errorbar(i,mu,yerr=[[mu-lo],[hi-mu]],fmt="D",color=c,capsize=3,ms=5,zorder=3)
    ax.set_xticks([0,1],["Chronos 2","TimesFM3"]);ax.set_xlim(-.4,1.4);ax.set_ylim(bottom=0)
    ax.set_ylabel("NMAE");ax.set_xlabel("Backbone within BiTFI");ps.panel(ax,"D","BiTFI backbone comparison")
    pd.DataFrame([dict(Chronos2_NMAE=av.mean(),TimesFM3_NMAE=bv.mean(),TFM3_reduction_percent=100*(1-bv.mean()/av.mean()),paired_wilcoxon_p=float(wilcoxon(av,bv).pvalue),n_greenhouses=len(av))]).to_csv(folder/"bitfi_backbone_comparison.csv",index=False)
    handles,labels=axes[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc="upper center",bbox_to_anchor=(.55,.985),fontsize=8,handlelength=2)
    ps.save(fig,"FigureS6_Univariate_backbone_comparison",supp=True)
    print(summary.to_string(index=False));print(tests.to_string(index=False))
if __name__=="__main__":main()

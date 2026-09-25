"""Redesign of F1_data_overview using its original _collect, split and sampling."""
import sys,json,contextlib,io
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import seaborn as sns
sys.path.insert(0,str(Path(__file__).resolve().parent))
import F1_data_overview as original
import prism_style as ps
ROOT=ps.ROOT;OUT=ps.OUT
ROLE={"train":"#6D9FAD","test":"#D99A79"}
VC=["#C99055","#76A9C6","#6DA58D","#AD83A8","#C5B66E"]
def main():
    ps.setup()
    plt.rcParams.update({"axes.linewidth":.8,"xtick.major.width":.8,"ytick.major.width":.8,"xtick.major.size":3.5,"ytick.major.size":3.5,"font.size":8,"xtick.labelsize":7,"ytick.labelsize":7})
    split=json.loads((ROOT/"03_result/comparison/split.json").read_text())
    with contextlib.redirect_stdout(io.StringIO()):recs,miss,dist,corr=original._collect(split)
    train=sorted([r for r in recs if r["role"]=="train"],key=lambda r:r["start"])
    test=sorted([r for r in recs if r["role"]=="test"],key=lambda r:r["start"])
    fig=plt.figure(figsize=(7.2,8.3))
    def title(x,y,letter,label):
        fig.text(x,y,letter,weight="bold",fontsize=11)
        fig.text(x+.035,y,label,weight="bold",fontsize=9)
    title(.065,.967,"A","Coverage across greenhouses")
    ax=fig.add_axes([.105,.59,.87,.35])
    ordered=train+test
    for i,r in enumerate(ordered):
        y=i+(1.2 if r["role"]=="test" else 0)
        ax.barh(y,mdates.date2num(r["end"])-mdates.date2num(r["start"]),left=mdates.date2num(r["start"]),height=.65,color=ROLE[r["role"]],edgecolor="none")
    ys=list(range(len(train)))+[i+len(train)+1.2 for i in range(len(test))]
    ax.set_yticks(ys,[f"Tr {i+1:02}" for i in range(len(train))]+[f"Te {i+1:02}" for i in range(len(test))],fontsize=5.8)
    ax.set_ylim(ys[-1]+.9,-.9);ax.tick_params(axis="y",length=0,pad=3)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(min(r["start"] for r in recs),max(r["end"] for r in recs))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.grid(axis="x",color="#E9ECEF",lw=.5);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=ROLE["train"],label=f"Training · {len(train)} sites"),Patch(facecolor=ROLE["test"],label=f"Test · {len(test)} sites")],
               loc="upper right",bbox_to_anchor=(.984,.989),ncol=2,fontsize=7,handlelength=1.1,columnspacing=1.1)
    title(.065,.535,"B","Environmental distributions")
    summaries=[]
    for j,v in enumerate(ps.VARS):
        ax=fig.add_axes([.105+j*.18,.337,.132,.16])
        vals=np.concatenate(dist[v])
        sns.violinplot(y=vals,ax=ax,color=VC[j],inner=None,cut=0,linewidth=.65,saturation=.8)
        q=np.quantile(vals,[.25,.5,.75])
        ax.plot([0,0],[q[0],q[2]],color="#303940",lw=2.5,zorder=3)
        ax.scatter([0],[q[1]],s=12,c="white",edgecolors="#303940",linewidths=.6,zorder=4)
        ax.set_xticks([]);ax.set_xlabel(original.fu.VAR_SHORT_LABELS[v],fontsize=8,labelpad=4)
        ax.set_ylabel(r"W m$^{-2}$" if v=="Rad" else ps.UNITS[v],fontsize=7,labelpad=3);ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        summaries.append(dict(variable=v,n_sampled=len(vals),Q1=q[0],median=q[1],Q3=q[2]))
    title(.065,.267,"C","Missingness across sites")
    ax=fig.add_axes([.105,.075,.39,.16]);rng=np.random.default_rng(42)
    for j,v in enumerate(ps.VARS):
        for k,role in enumerate(["train","test"]):
            values=miss.loc[miss.role==role,v].dropna().to_numpy()*100
            x=j+(k-.5)*.32
            ax.bar(x,values.mean(),width=.28,color=ROLE[role],alpha=.3,lw=0,zorder=1)
            ax.scatter(x+rng.uniform(-.065,.065,len(values)),values,s=7,facecolors="white",edgecolors=ROLE[role],lw=.6,alpha=.8,zorder=2)
            ax.plot([x-.12,x+.12],[values.mean()]*2,color=ROLE[role],lw=1.8,zorder=3)
    ax.set_xticks(range(5),[original.fu.VAR_SHORT_LABELS[v] for v in ps.VARS]);ax.set_ylabel("Missing observations (%)",fontsize=7);ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    title(.565,.267,"D","Inter-variable correlation")
    ax=fig.add_axes([.62,.075,.29,.16]);mat=corr.loc[ps.VARS,ps.VARS].to_numpy()
    cmap=sns.diverging_palette(235,15,s=65,l=55,as_cmap=True)
    im=ax.imshow(mat,vmin=-1,vmax=1,cmap=cmap,aspect="auto")
    labels=[original.fu.VAR_SHORT_LABELS[v] for v in ps.VARS]
    ax.set_xticks(range(5),labels);ax.set_yticks(range(5),labels);ax.tick_params(length=0,pad=3)
    for sp in ax.spines.values():sp.set_visible(False)
    for i in range(5):
        for j in range(5):ax.text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",fontsize=6.5,color="white" if abs(mat[i,j])>.7 else "#303940")
    cax=fig.add_axes([.927,.075,.012,.16]);cb=fig.colorbar(im,cax=cax,ticks=[-1,0,1]);cb.outline.set_visible(False);cb.ax.tick_params(width=.8,length=2,labelsize=6.5);cb.set_label("Pearson r",fontsize=7,labelpad=3)
    fig.savefig(OUT/"Figure1_Dataset_profile.pdf")
    plt.close(fig)
    folder=OUT/"source_data/Figure1_profile";folder.mkdir(exist_ok=True)
    miss.to_csv(folder/"missingness.csv",index=False);corr.to_csv(folder/"correlation.csv");pd.DataFrame(summaries).to_csv(folder/"distribution_summary.csv",index=False)
    pd.DataFrame([{k:r[k] for k in ["name","role","start","end"]} for r in ordered]).to_csv(folder/"coverage.csv",index=False)
    caption="""Fig. 1. Greenhouse dataset profile. (A) Recording spans of the training and test greenhouses. (B) Variable distributions, with medians and interquartile ranges. (C) Missingness by variable and dataset split. (D) Mean within-greenhouse Pearson correlations.
"""
    (OUT/"Figure1_Dataset_profile_caption.md").write_text(caption)
    print("Sites:",len(train),len(test));print("Correlation:",np.round(mat,2));print("Original missingness group means (%):",miss.groupby("role")[ps.VARS].mean()*100)
if __name__=="__main__":main()

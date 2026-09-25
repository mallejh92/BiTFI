"""Training-site validation context selection; test data are not used."""
import json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import prism_style as ps

def main():
 root=ps.ROOT/'03_result/context_validation_20260925';summary=pd.read_csv(root/'summary.csv');g=pd.read_csv(root/'greenhouse_scores.csv');selected=json.loads((root/'selected_contexts.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
 ps.setup();fig,axes=plt.subplots(1,2,figsize=(7.2,3.5));fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.79,wspace=.36)
 labels={'Chronos2':'Chronos 2','TimesFM2.5':'TimesFM2.5','TimesFM3.0':'TimesFM3'}
 for m,lab in labels.items():
  d=summary[summary.model==m].sort_values('context');c=ps.COLORS[m];axes[0].plot(d.context,d['mean'],color=c,marker='o',ms=3,label=lab)
  best=d[d.context==selected[m]].iloc[0];axes[0].scatter(best.context,best['mean'],s=55,facecolor='white',edgecolor=c,lw=1.4,zorder=4)
  relative=d['mean']/best['mean']-1;axes[1].plot(d.context,100*relative,color=c,marker='o',ms=3)
 for ax in axes:
  ax.set_xticks([96,720,1440,1900]);ax.set_xlabel('Context length (h)')
 axes[0].set_ylabel('Validation NMAE');axes[1].set_ylabel('Error above selected minimum (%)');axes[1].set_ylim(bottom=-.2)
 ps.panel(axes[0],'A','Univariate validation');ps.panel(axes[1],'B','Sensitivity within backbone')
 h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='upper center',bbox_to_anchor=(.54,1.0))
 desc='Selected: '+ '  |  '.join(f'{labels[m]} {selected[m]:,} h' for m in labels)
 fig.text(.54,.02,desc,ha='center',fontsize=7.5)
 ps.save(fig,'FigureS2_Context_sensitivity',supp=True)
 dest=ps.OUT/'source_data';dest.mkdir(exist_ok=True)
 summary.to_csv(dest/'context_validation_summary.csv',index=False);g.to_csv(dest/'context_validation_greenhouse_scores.csv',index=False)
 (dest/'context_validation_protocol.json').write_text(json.dumps(protocol,indent=2));(dest/'selected_contexts.json').write_text(json.dumps(selected,indent=2))
if __name__=='__main__':main()

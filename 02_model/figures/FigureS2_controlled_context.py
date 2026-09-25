"""Matched-input S2: three backbones, two information arms, independent y scales."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import prism_style as ps
MODELS=['Chronos2','TimesFM2.5','TimesFM3.0'];ARMS=['univariate','local_covariates'];CONTEXTS=[96,168,336,720,1080,1440,1900]
def main():
 root=ps.RESULT_ROOT/'controlled_context_sweep_20260922';frames=[]
 for m in MODELS:
  assert (root/f'{m}_complete.json').exists(),m
  for arm in ARMS:
   for c in CONTEXTS:
    d=pd.read_csv(root/f'{m}_{arm}_ctx{c}.csv');assert len(d)==299 and not d.case_id.duplicated().any()
    assert np.isfinite(d.NMAE).all();frames.append(d)
 full=pd.concat(frames,ignore_index=True)
 reference=None
 for _,d in full.groupby(['model','arm','context']):
  ids=set(d.case_id)
  if reference is None:reference=ids
  assert ids==reference
 z=full.assign(w=full.NMAE*full.n_eval).groupby(['model','arm','context','greenhouse'])[['w','n_eval']].sum()
 scores=(z.w/z.n_eval).rename('NMAE').reset_index();summary=scores.groupby(['model','arm','context']).NMAE.agg(['mean','std','count']).reset_index()
 dest=ps.OUT/'source_data';dest.mkdir(exist_ok=True)
 full.to_csv(dest/'context_controlled_case_scores.csv',index=False);scores.to_csv(dest/'context_sweep_greenhouse_scores.csv',index=False);summary.to_csv(dest/'context_controlled_summary.csv',index=False)
 values=summary['mean'];span=values.max()-values.min();step=.05 if span>.1 else (.02 if span>.05 else .005);lo=max(0.,np.floor((values.min()-.001)/step)*step);hi=np.ceil((values.max()+.001)/step)*step
 ps.setup();fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),sharey=False);fig.subplots_adjust(left=.10,right=.97,wspace=.24,bottom=.20,top=.80)
 labels={'Chronos2':'Chronos 2','TimesFM2.5':'TimesFM2.5','TimesFM3.0':'TimesFM3'}
 for ax,arm,letter,title in zip(axes,ARMS,'AB',['Univariate','Local covariates']):
  for m in MODELS:
   d=summary[(summary.model==m)&(summary.arm==arm)].sort_values('context')
   ax.plot(d.context,d['mean'],color=ps.COLORS[m],marker='o',label=labels[m])
  if arm == "local_covariates":
   ax.set_ylim(lo,hi);ax.set_yticks(np.arange(lo,hi+step/2,step))
  ax.tick_params(labelleft=True)
  ax.set_xticks([96,720,1440,1900]);ax.set_xlabel('Context length (h)');ax.set_ylabel('NMAE');ax.set_title(title,loc='left',pad=10);ax.text(-.16,1.16,letter,transform=ax.transAxes,fontweight='bold',fontsize=13,va='top')
 h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc='upper center',bbox_to_anchor=(.54,1.01),frameon=False)
 ps.save(fig,'FigureS2_Context_sensitivity',supp=True)
 (dest/'context_controlled_validation.json').write_text(json.dumps(dict(configurations=42,cases_per_configuration=299,matched_case_ids=True,panel_ylim={"A":list(axes[0].get_ylim()),"B":list(axes[1].get_ylim())},A_autoscale=True,feature_poison_checks=2093),indent=2))
 print(summary.to_string(index=False))
if __name__=='__main__':main()

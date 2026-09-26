"""Validation selection, test depth sensitivity and A6000 throughput in Figure S8."""
from pathlib import Path
import sys,json,shutil
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'02_model/figures'))
import prism_style as ps
R=ROOT/'03_result/revision_experiments_20260926/analysis';V=ROOT/'03_result/refinement_validation_20260926';steps=[0,1,2,3,5]
def main():
 k=json.loads((V/'selection.json').read_text())['selected_refinements'];va=pd.read_csv(V/'summary.csv').set_index('refinements');te=pd.read_csv(R/'refinement_summary.csv').set_index('model');variables=pd.read_csv(R/'refinement_variables.csv');timing=json.loads((V/'a6000_main_test_timing.json').read_text())['seconds_per_mask_by_refinements']
 ps.setup();fig,axs=plt.subplots(2,2,figsize=(7.2,5.9));fig.subplots_adjust(left=.10,right=.98,bottom=.12,top=.89,wspace=.37,hspace=.58)
 for ax,d,title,letter in [(axs[0,0],va,'Training-site validation','A'),(axs[0,1],te,'Test-site depth sensitivity','B')]:
  for j,step in enumerate(steps):
   row=d.loc[step if letter=='A' else f'BiTFI-refine{step}'];c=ps.COLORS['DAFI-TimesFM3'] if step==k else '#71859D'
   ax.errorbar(j,row['mean'],yerr=[[row['mean']-row.low],[row.high-row['mean']]],fmt='D' if step==k else 'o',color=c,capsize=3,ms=4)
  ax.set_xticks(range(5),['0','1','2','3','5']);ax.set_xlabel('Refinement passes');ax.set_ylabel('NMAE');ps.panel(ax,letter,title)
 ax=axs[1,0]
 for step,c,label in [(0,'#71859D','Step 1 only'),(k,ps.COLORS['DAFI-TimesFM3'],'BiTFI')]:
  g=variables[variables.model==f'BiTFI-refine{step}'].set_index('variable').reindex(ps.VARS);ax.plot(range(5),g['mean'],marker='o',ms=4,color=c,label=label)
 ax.set_xticks(range(5),['Tin','Tout','RH','CO₂','Rad']);ax.set_ylabel('Test NMAE');ax.legend(fontsize=7);ps.panel(ax,'C','Variable-specific contribution')
 ax=axs[1,1];ax.plot(steps,[timing[str(s)] for s in steps],color='#71859D',marker='o',ms=4);ax.scatter([k],[timing[str(k)]],color=ps.COLORS['DAFI-TimesFM3'],marker='D',s=28,zorder=3);ax.set_xticks(steps);ax.set_xlabel('Refinement passes');ax.set_ylabel('Seconds per mask job');ax.set_ylim(bottom=0);ps.panel(ax,'D','RTX A6000 batched throughput')
 ps.save(fig,'FigureS8_Refinement_ablation',supp=True);shutil.copy2(ps.OUT/'Supplementary/FigureS8_Refinement_ablation.pdf',ROOT/'05_thesis/FigureS8_Refinement_ablation.pdf')
 for name in ['summary.csv','greenhouse_scores.csv','selection.json','a6000_main_test_timing.json']:shutil.copy2(V/name,ROOT/'04_figure/source_data'/('refinement_validation_'+name))
 print('Rendered validation-selected depth',k)
if __name__=='__main__':main()

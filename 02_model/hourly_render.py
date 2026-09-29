"""Render all manuscript figures from the completed hourly experiment."""
import os,json,sys,shutil
from pathlib import Path
import clean_protocol as cp
ROOT=cp.ROOT;OUT=cp.OUT;AN=OUT/'analysis'
os.environ['BITFI_CLEAN']='1';os.environ['BITFI_RESULT_ROOT']=str(OUT)
sys.path.insert(0,str(ROOT/'02_model/figures'))
import numpy as np,pandas as pd,matplotlib.pyplot as plt
import prism_style as ps
import build_prism_figures as b

def main(results_only=False):
 ps.setup();b.DATA.mkdir(parents=True,exist_ok=True)
 protocol=json.loads((OUT/'protocol.json').read_text());protocol['selected_contexts']=json.loads((OUT/'context_validation/selected_contexts.json').read_text());protocol['selected_refinements']=json.loads((OUT/'refinement_validation/selection.json').read_text())['selected_refinements'];(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 full=pd.read_csv(AN/'comparison_results.csv');main=full[full.model.isin(ps.MAIN)]
 from Figure1_dataset_profile import main as f1
 from Figure2_BiTFI_framework import main as f2
 if not results_only:f1();f2()
 b.overall(full);b.gap_robustness(full);b.variables(full);b.examples();b.supplementary(full,main)
 from FigureS7_univariate_backbones import main as s6
 from FigureS8_moment_tuning import main as s7
 s6();s7()
 depth=protocol['selected_refinements'];steps=[0,1,2,3,5];va=pd.read_csv(OUT/'refinement_validation/summary.csv').set_index('refinements');te=pd.read_csv(AN/'refinement_summary.csv').set_index('model');variables=pd.read_csv(AN/'refinement_variables.csv');timing=json.loads((AN/'a6000_timing.json').read_text())['seconds_per_mask_by_refinements']
 fig,axs=plt.subplots(2,2,figsize=(7.2,5.9));fig.subplots_adjust(left=.10,right=.98,bottom=.12,top=.89,wspace=.37,hspace=.58)
 for ax,d,title,letter in [(axs[0,0],va,'Training-site validation','A'),(axs[0,1],te,'Test-site depth sensitivity','B')]:
  for j,k in enumerate(steps):
   r=d.loc[k if letter=='A' else f'BiTFI-refine{k}'];c=ps.COLORS['BiTFI-TimesFM3'] if k==depth else '#71859D';ax.errorbar(j,r['mean'],yerr=[[r['mean']-r.low],[r.high-r['mean']]],fmt='D' if k==depth else 'o',color=c,capsize=3,ms=4)
  ax.set_xticks(range(5),[str(k) for k in steps]);ax.set_xlabel('Refinement passes');ax.set_ylabel('NMAE');ps.panel(ax,letter,title)
 ax=axs[1,0]
 for k,c,label in [(0,'#71859D','Step 1 only'),(depth,ps.COLORS['BiTFI-TimesFM3'],'BiTFI')]:
  q=variables[variables.model==f'BiTFI-refine{k}'].set_index('variable').reindex(ps.VARS);ax.plot(range(5),q['mean'],marker='o',ms=4,color=c,label=label)
 ax.set_xticks(range(5),['Tin','Tout','RH','CO₂','Rad']);ax.set_ylabel('Test NMAE');ax.legend(fontsize=7.5);ps.panel(ax,'C','Variable-specific contribution')
 ax=axs[1,1];ax.plot(steps,[timing[str(k)] for k in steps],color='#71859D',marker='o',ms=4);ax.scatter([depth],[timing[str(depth)]],color=ps.COLORS['BiTFI-TimesFM3'],marker='D',s=28,zorder=3);ax.set_xticks(steps);ax.set_xlabel('Refinement passes');ax.set_ylabel('Seconds per mask job');ax.set_ylim(bottom=0);ps.panel(ax,'D','RTX A6000 inference time');ps.save(fig,'FigureS8_Refinement_ablation',supp=True)
 fig,axs=plt.subplots(1,2,figsize=(7.2,3.6));fig.subplots_adjust(left=.10,right=.98,wspace=.36,bottom=.30,top=.82)
 for ax,file,models,labels,title,letter in [(axs[0],'all_summary.csv',['SAITS','SAITS-matched-local','SAITS-spatial'],['Sensors only','Local','Local + cross'],'SAITS input configurations','A'),(axs[1],'additional_summary.csv',['LI','Spatial-Ridge','TimesFM3-univariate','BiTFI'],['Linear\ninterpolation','Spatial\nridge','TimesFM3\n(univariate)','BiTFI'],'Nine additional greenhouses','B')]:
  q=pd.read_csv(AN/file).set_index('model')
  for j,m in enumerate(models):
   r=q.loc[m];ax.errorbar(j,r['mean'],yerr=[[r['mean']-r.low],[r.high-r['mean']]],fmt='D',color=ps.COLORS.get(m,{'TimesFM3-univariate':'#397DB8','BiTFI':'#C84E69'}.get(m,'#71859D')),capsize=3)
  ax.set_xticks(range(len(models)),labels,fontsize=8);ax.set_xlim(-.5,len(models)-.5);ax.set_ylim(bottom=0);ax.set_ylabel('NMAE');ps.panel(ax,letter,title)
 ps.save(fig,'FigureS9_Information_and_additional_sites',supp=True)
 # Residual strata are descriptive, with thresholds fixed using training prefixes.
 q=pd.read_csv(AN/'residual_strata_MAE.csv');q=q[q.model.isin(['BiTFI-TimesFM3','TimesFM3.0-COV-SPA'])];fig,axs=plt.subplots(2,3,figsize=(7.2,5.6));fig.subplots_adjust(left=.10,right=.98,bottom=.13,top=.90,wspace=.47,hspace=.72)
 for i,ax in enumerate(axs.flat):
  v=ps.VARS[i] if i<5 else 'Rad';strata=['change_low','change_high'] if i<5 else ['radiation_zero','radiation_positive'];labels=['Lower change','Higher change'] if i<5 else ['Zero','Positive']
  for j,m in enumerate(['TimesFM3.0-COV-SPA','BiTFI-TimesFM3']):
   z=q[(q.variable==v)&(q.model==m)].set_index('stratum').loc[strata];x=np.arange(2)+(j-.5)*.12;ax.errorbar(x,z['mean'],yerr=[z['mean']-z.low,z.high-z['mean']],color=ps.COLORS[m],fmt='o-',capsize=2,ms=3,label='BiTFI' if m.startswith('BiTFI') else 'TimesFM3')
  ax.set_xticks(range(2),labels,fontsize=7.5);ax.set_xlim(-.3,1.3);ax.set_ylim(bottom=0);ax.set_ylabel('MAE ('+('percentage points' if v=='RH' else ps.UNITS[v])+')',fontsize=8);ps.panel(ax,chr(65+i),ps.VL[v] if i<5 else 'Irradiance state');ax.texts[-1].set_position((-.15,1.15))
 h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=2,loc='upper center',bbox_to_anchor=(.55,1.015),fontsize=8);ps.save(fig,'FigureS10_Environmental_transitions',supp=True)
 for p in AN.glob('*.csv'):
  if not p.name.startswith('early_'):shutil.copy2(p,b.DATA/p.name)
 for p in AN.glob('*.json'):shutil.copy2(p,b.DATA/p.name)
 print('Rendered all hourly figures',ps.OUT,flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--results-only',action='store_true');args=parser.parse_args();main(results_only=args.results_only)

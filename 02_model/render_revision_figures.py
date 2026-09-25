"""Render the common comparison set and supplementary revision diagnostics."""
import os,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'03_result/revision_experiments_20260926';AN=RUN/'analysis'
os.environ['BITFI_RESULT_ROOT']=str(ROOT/'03_result/reevaluation_context_20260925');sys.path[:0]=[str(ROOT/'02_model/figures')]
import numpy as np,pandas as pd,matplotlib.pyplot as plt
import build_prism_figures as b
from analyze_revision_experiments import site_scores,summary
from revision_config import MAIN_MODELS
ps=b.ps;ps.setup();b.DATA.mkdir(parents=True,exist_ok=True)
full,main=b.load_results();full=full[~full.model.str.startswith('CAFI')];full.to_csv(AN/'comparison_results.csv',index=False)
b.overall(full);b.gap_robustness(full);b.variables(full);b.examples();b.supplementary(full,main)
summary(full.query("group_type=='all'")).to_csv(AN/'all_summary.csv',index=False)
summary(full.query("group_type=='all'"),extra=['variable']).to_csv(AN/'all_variables.csv',index=False)
summary(full.query("group_type=='all'"),'MAE',['variable']).to_csv(AN/'all_physical_MAE.csv',index=False)
summary(full.query("group_type=='season'"),extra=['group_value']).to_csv(AN/'all_seasons.csv',index=False)
# Main figures can be reviewed while the independent ablation job runs.
for p in (ROOT/'04_figure').glob('Figure*.pdf'):shutil.copy2(p,ROOT/'05_thesis'/p.name)
for p in (ROOT/'04_figure/Supplementary').glob('Figure*.pdf'):shutil.copy2(p,ROOT/'05_thesis'/p.name)
if '--main-only' in sys.argv:print('Rendered main comparison figures');raise SystemExit(0)
# Two panels for algorithmic contribution, without selecting the best pass on test data.
d=pd.read_csv(AN/'refinement_summary.csv');v=pd.read_csv(AN/'refinement_variables.csv');fig,ax=plt.subplots(1,2,figsize=(7.2,3.25));fig.subplots_adjust(left=.10,right=.98,wspace=.40,bottom=.21,top=.83)
steps=[0,1,2,3,5]
for j,step in enumerate(steps):
 r=d[d.model=='BiTFI-refine'+str(step)].iloc[0];c=ps.COLORS['DAFI-TimesFM3'] if step==1 else '#71859D';ax[0].errorbar(j,r['mean'],yerr=[[r['mean']-r.low],[r.high-r['mean']]],fmt='o',color=c,capsize=3)
ax[0].set_xticks(range(5),['0\nStep 1','1\nBiTFI','2','3','5']);ax[0].set_xlabel('Covariate refinement passes');ax[0].set_ylabel('NMAE');ps.panel(ax[0],'A','Refinement depth')
for step,c,label in [(0,'#71859D','Step 1 only'),(1,ps.COLORS['DAFI-TimesFM3'],'BiTFI')]:
 g=v[v.model=='BiTFI-refine'+str(step)].set_index('variable').reindex(ps.VARS);ax[1].plot(range(5),g['mean'],marker='o',color=c,label=label)
ax[1].set_xticks(range(5),['Tin','Tout','RH','CO₂','Rad']);ax[1].set_ylabel('NMAE');ax[1].legend(fontsize=8);ps.panel(ax[1],'B','Initialization and full reconstruction');ps.save(fig,'FigureS8_Refinement_ablation',supp=True)
# Matched information controls and additional sites.
fig,ax=plt.subplots(1,2,figsize=(7.2,3.6));fig.subplots_adjust(left=.10,right=.98,wspace=.36,bottom=.30,top=.82)
for a,file,models,labels,title,letter in [(ax[0],'saits_summary.csv',['SAITS-matched-local','SAITS-spatial'],['SAITS-local','SAITS-cross'],'Reference-site information','A'),(ax[1],'additional_summary.csv',['LI','Spatial-Ridge','TimesFM3-univariate','BiTFI'],['Linear\ninterpolation','Spatial\nridge','TimesFM3\n(univariate)','BiTFI'],'Nine additional greenhouses','B')]:
 q=pd.read_csv(AN/file)
 for j,m in enumerate(models):
  r=q[q.model==m].iloc[0];a.errorbar(j,r['mean'],yerr=[[r['mean']-r.low],[r.high-r['mean']]],fmt='D',color=ps.COLORS.get(m,{'TimesFM3-univariate':'#397DB8','BiTFI':'#C84E69'}.get(m,'#71859D')),capsize=3)
 a.set_xticks(range(len(models)),labels,fontsize=8);a.set_xlim(-.5,len(models)-.5);a.set_ylim(bottom=0);a.set_ylabel('NMAE');ps.panel(a,letter,title)
ps.save(fig,'FigureS9_Information_and_additional_sites',supp=True)
# Agronomically interpretable physical metrics, main Fig. 7.
g=pd.read_csv(AN/'agronomic_by_gap.csv');fig,ax=plt.subplots(1,2,figsize=(7.2,3.7));fig.subplots_adjust(left=.11,right=.98,wspace=.37,bottom=.18,top=.70)
models=['LI','SAITS-matched-local','SAITS-spatial','BiTFI-refine0','BiTFI-refine1'];names=['Linear interpolation','SAITS-local','SAITS-cross','Step 1 only','BiTFI (TimesFM3)'];colors=['#939AA3','#9B75BE','#503879','#71859D','#C84E69']
for a,metric,letter,title,unit in [(ax[0],'VPD','A','Air VPD','MAE (kPa)'),(ax[1],'radiation_integral','B','Outdoor radiation integral','Absolute error (MJ m⁻²)')]:
 for m,label,c in zip(models,names,colors):
  q=g[(g.model==m)&(g.metric==metric)].set_index('gap_length_h').reindex([6,12,24,72,168]);a.plot(range(5),q['mean'],marker='o',color=c,label=label,ms=4,lw=1.6)
 a.set_xticks(range(5),[6,12,24,72,168]);a.set_xlabel('Gap length (h)');a.set_ylabel(unit)
 if metric=='radiation_integral':
  from matplotlib.ticker import ScalarFormatter
  a.set_yscale('log');a.set_yticks([.1,1,10,100]);a.yaxis.set_major_formatter(ScalarFormatter());a.minorticks_off()
 else:a.set_ylim(bottom=0)
 ps.panel(a,letter,title)
fig.legend(*ax[0].get_legend_handles_labels(),ncol=3,loc='upper center',bbox_to_anchor=(.54,1),fontsize=8);ps.save(fig,'Figure7_Agronomic_diagnostics')
# Low-light versus illuminated periods are diagnostic strata, not a day/night assumption.
g=pd.read_csv(AN/'illumination_summary.csv');g=g[g.model=='BiTFI-refine1'];fig,axs=plt.subplots(1,5,figsize=(7.2,2.85));fig.subplots_adjust(left=.08,right=.99,wspace=.65,bottom=.27,top=.78)
for j,(a,v) in enumerate(zip(axs,ps.VARS)):
 for k,period in enumerate(['low-light','light']):
  r=g[(g.variable==v)&(g.period==period)].iloc[0];a.errorbar(k,r['mean'],yerr=[[r['mean']-r.low],[r.high-r['mean']]],fmt='o',color=['#71859D','#C84E69'][k],capsize=3)
 a.set_xticks([0,1],['≤20','>20']);a.set_xlim(-.5,1.5);a.set_ylim(bottom=0);a.set_xlabel('Radiation\n(W m⁻²)',fontsize=8);a.set_ylabel('MAE ('+ps.UNITS[v]+')',fontsize=8);ps.panel(a,chr(65+j),v)
ps.save(fig,'FigureS10_Illumination_diagnostics',supp=True)
# Publish only actual final PDF assets; source data remain auditable.
for p in (ROOT/'04_figure').glob('Figure*.pdf'):shutil.copy2(p,ROOT/'05_thesis'/p.name)
for p in (ROOT/'04_figure/Supplementary').glob('Figure*.pdf'):shutil.copy2(p,ROOT/'05_thesis'/p.name)
print('Rendered revised figures')

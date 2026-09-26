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
for a,file,models,labels,title,letter in [(ax[0],'all_summary.csv',['SAITS','SAITS-matched-local','SAITS-spatial'],['Sensors only','Local','Local + cross'],'SAITS input configurations','A'),(ax[1],'additional_summary.csv',['LI','Spatial-Ridge','TimesFM3-univariate','BiTFI'],['Linear\ninterpolation','Spatial\nridge','TimesFM3\n(univariate)','BiTFI'],'Nine additional greenhouses','B')]:
 q=pd.read_csv(AN/file)
 for j,m in enumerate(models):
  r=q[q.model==m].iloc[0];a.errorbar(j,r['mean'],yerr=[[r['mean']-r.low],[r.high-r['mean']]],fmt='D',color=ps.COLORS.get(m,{'TimesFM3-univariate':'#397DB8','BiTFI':'#C84E69'}.get(m,'#71859D')),capsize=3)
 a.set_xticks(range(len(models)),labels,fontsize=8);a.set_xlim(-.5,len(models)-.5);a.set_ylim(bottom=0);a.set_ylabel('NMAE');ps.panel(a,letter,title)
ps.save(fig,'FigureS9_Information_and_additional_sites',supp=True)
# Publish only actual final PDF assets; source data remain auditable.
for p in (ROOT/'04_figure').glob('Figure*.pdf'):shutil.copy2(p,ROOT/'05_thesis'/p.name)
for p in (ROOT/'04_figure/Supplementary').glob('Figure*.pdf'):shutil.copy2(p,ROOT/'05_thesis'/p.name)
print('Rendered revised figures')

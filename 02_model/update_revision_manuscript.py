"""Refresh publication tables from executed results and the shared display configuration.

The manuscript files are local publication inputs and are not included in the
public code/data repository. This utility is idempotent and does not infer results.
"""
from pathlib import Path
import re
import pandas as pd
from revision_config import MAIN_MODELS,LABELS
from figures.prism_style import LABELS as ALL_LABELS
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'05_thesis';A=ROOT/'03_result/revision_experiments_20260926/analysis';D=ROOT/'04_figure/source_data'
labels=ALL_LABELS|LABELS
def table(caption,label,heads,rows,cols=None):
 cols=cols or ('l'+'r'*(len(heads)-1));body='\n'.join(' & '.join(row)+r' \\' for row in rows)
 return '\n'.join([r'\begin{table}[p]',r'\centering\fontfamily{ptm}\selectfont\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\setlength{\tabcolsep}{6pt}',r'\begin{tabular}{@{}'+cols+'@{}}',r'\toprule',' & '.join(heads)+r' \\',r'\midrule',body,r'\bottomrule',r'\end{tabular}',r'\end{table}'])
def replace_table(s,label,new):
 pattern=r'\\begin\{table\*?\}(?:(?!\\end\{table).)*?\\label\{'+re.escape(label)+r'\}.*?\\end\{table\*?\}'
 s,n=re.subn(pattern,lambda m:new,s,flags=re.S);assert n==1,(label,n);return s

def f(x):return f'{x:.4f}'
def ci(r):return '['+f(r.low)+', '+f(r.high)+']'
def read(n):return pd.read_csv(A/(n+'.csv'))
def main():
 s=(T/'TFM.tex').read_text();su=(T/'supplementary.tex').read_text()
 allr=read('all_summary').set_index('model');var=read('all_variables');seas=read('all_seasons');physical=read('all_physical_MAE');tests=pd.read_csv(D/'paired_greenhouse_tests.csv');n=len(MAIN_MODELS)
 pv=tests.set_index('reference').loc['TimesFM3.0-COV-SPA','holm_p']
 for name in ['s','su']:
  text=s if name=='s' else su
  text=re.sub(r'\\newcommand\{\\MainHolmP\}\{[^}]*\}',lambda m:r'\newcommand{\MainHolmP}{'+f(pv)+'}',text)
  if name=='s':s=text
  else:su=text
 rows=[[labels[m],f(allr.loc[m,'mean']),f(allr.loc[m,'sd']),ci(allr.loc[m])] for m in sorted(MAIN_MODELS,key=lambda m:allr.loc[m,'mean'])]
 s=replace_table(s,'tab:overall',table(fr'Overall reconstruction error for the {n} representative configurations across 10 test greenhouses. SD is between-greenhouse standard deviation; CI is the 95\% bootstrap confidence interval.','tab:overall',['Configuration','NMAE','SD',r'95\% CI'],rows))
 rows=[]
 for m in MAIN_MODELS:
  values=[]
  for v in ['Tin','Tout','RH','CO2','Rad']:
   value=var[(var.model==m)&(var.variable==v)].iloc[0]['mean'];txt=f(value)
   if value==var[var.model.isin(MAIN_MODELS)&(var.variable==v)]['mean'].min():txt=r'\textbf{'+txt+'}'
   values.append(txt)
  rows.append([labels[m],*values])
 s=replace_table(s,'tab:variable',table('Variable-specific mean greenhouse NMAE for the representative configurations. Bold values indicate the lowest error in each column; cross denotes cross-greenhouse covariates.','tab:variable',['Model',r'$T_{\mathrm{in}}$',r'$T_{\mathrm{out}}$','RH',r'CO$_2$','Rad'],rows))
 rows=[[labels[m],f(r['mean']),f(r.sd)] for m,r in allr.sort_values('mean').iterrows()]
 su=replace_table(su,'tab:all',table('Mean greenhouse NMAE and between-greenhouse SD for all 21 comparison settings, ordered by NMAE. Step-depth diagnostics are listed separately in Table~S4.','tab:all',['Configuration','NMAE','SD'],rows))
 rows=[[labels[r.reference],f'{r.reduction_percent:.2f}',f'[{r.CI_low:.2f}, {r.CI_high:.2f}]',f(r.wilcoxon_p),f(r.holm_p)] for r in tests.itertuples()]
 su=replace_table(su,'tab:paired',table(fr'Paired comparisons of BiTFI with the {n-1} references in the main aggregate comparison. Positive reductions favor BiTFI; intervals are 95\% paired bootstrap confidence intervals.','tab:paired',['Reference',r'Reduction (\%)',r'95\% CI (\%)','$p$','Holm $p$'],rows))
 rows=[[labels[m],*[f(seas[(seas.model==m)&(seas.group_value==season)].iloc[0]['mean']) for season in ['spring','summer','fall','winter']]] for m in MAIN_MODELS]
 su=replace_table(su,'tab:season',table('Seasonal mean greenhouse NMAE for the representative configurations.','tab:season',['Configuration','Spring','Summer','Fall','Winter'],rows))
 rows=[[labels[m],f(allr.loc[m,'mean']),f(allr.loc[m,'sd']),ci(allr.loc[m])] for m in ['SAITS','SAITS-matched-local','SAITS-spatial']]
 su=replace_table(su,'tab:saits_info',table(r'SAITS input configurations. Sensors only uses the original five-channel checkpoint; local and local + cross share the 24-channel architecture, with reference channels available only in the latter. CI is the 95\% greenhouse bootstrap confidence interval.','tab:saits_info',['Configuration','NMAE','SD',r'95\% CI'],rows))
 rows=[[labels[m],*[f(physical[(physical.model==m)&(physical.variable==v)].iloc[0]['mean']) for v in ['Tin','Tout','RH','CO2','Rad']]] for m in MAIN_MODELS]
 su=replace_table(su,'tab:physical',table('Physical-unit MAE for the representative sensor-reconstruction configurations. RH errors are percentage points.','tab:physical',['Setting',r'$T_{\mathrm{in}}$ ($^{\circ}$C)',r'$T_{\mathrm{out}}$ ($^{\circ}$C)','RH (pp)',r'CO$_2$ (ppm)',r'Rad (W\,m$^{-2}$)'],rows))
 metrics=pd.read_csv(D/'figure6_panel_metrics.csv');rows=[]
 for panel,g in metrics.groupby('panel',sort=True):
  bit=g[g.model=='DAFI-TimesFM3'].iloc[0];best=g[g.model!='DAFI-TimesFM3'].sort_values('MAE').iloc[0];label=labels[best.model]
  if ' (' in label:label=r'\shortstack[l]{'+label.replace(' (',r'\\(')+'}'
  rows.append([panel,bit.variable,f(bit.R2),f(bit.MAE),label,f(best.R2),f(best.MAE)])
 su=replace_table(su,'tab:example',table('Panel scores for main-text Fig.~6. Each baseline is selected by the lowest gap-only MAE among Spatial ridge, head-tuned MOMENT and SAITS (local + cross); MAE uses the variable\'s physical unit. G and J are unmasked.','tab:example',['Panel','Variable',r'BiTFI $R^2$','BiTFI MAE','Best baseline',r'Baseline $R^2$','Baseline MAE'],rows,cols='llrrlrr'))
 (T/'TFM.tex').write_text(s);(T/'supplementary.tex').write_text(su);print('Refreshed tables for',n,'main configurations; Holm p =',f(pv))
if __name__=='__main__':main()

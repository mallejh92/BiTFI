"""Refresh numerical table bodies without replacing the author's manuscript prose."""
from pathlib import Path
import re,json
import pandas as pd
import clean_protocol as cp
from revision_config import MAIN_MODELS,LABELS
from build_hourly_manuscript import SHORT

ROOT=cp.ROOT;R=cp.OUT;A=R/'analysis'
def replace_tabular(text,label,tabular):
 pattern=r'\\begin\{table\*?\}.*?\\end\{table\*?\}'
 blocks=[m for m in re.finditer(pattern,text,re.S) if r'\label{'+label+'}' in m.group()]
 assert len(blocks)==1,(label,len(blocks))
 match=blocks[0];block=match.group();old=re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}',block,re.S).group()
 return text[:match.start()]+block.replace(old,tabular,1)+text[match.end():]
def table(headers,rows,align):
 return '\\begin{tabular}{@{}'+align+'@{}}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}'
def main():
 # Template output is a staging artifact; only its numerical tabular environments
 # are transferred into the current author-edited supplementary document.
 import build_hourly_supplement
 build_hourly_supplement.main()
 generated=(R/'publication/supplementary.tex').read_text()
 (R/'verification_20260929/generated_tables.tex').write_text(generated)
 path=ROOT/'05_thesis/supplementary.tex';text=path.read_text();updated=[]
 for block in re.findall(r'\\begin\{table\}.*?\\end\{table\}',generated,re.S):
  label=re.search(r'\\label\{([^}]+)\}',block).group(1);tab=re.search(r'\\begin\{tabular\}.*?\\end\{tabular\}',block,re.S).group()
  text=replace_tabular(text,label,tab);updated.append(label)
 assert len(updated)==14
 fragment=(R/'verification_20260929/sensitivity/supplementary_sensitivity_table.tex').read_text()
 if r'\label{tab:verified_sensitivity}' in text:
  block=next(b for b in re.findall(r'\\begin\{table\}.*?\\end\{table\}',text,re.S) if r'\label{tab:verified_sensitivity}' in b);text=text.replace(block,fragment.strip(),1)
 else:
  anchor=r'\begin{figure}'
  index=text.index(anchor);text=text[:index]+fragment+'\n\\clearpage\n\n'+text[index:]
 conditions=(R/'verification_20260929/comparison_conditions_table_s16.tex').read_text()
 if r'\label{tab:comparison_conditions}' in text:
  block=next(b for b in re.findall(r'\\begin\{table\}.*?\\end\{table\}',text,re.S) if r'\label{tab:comparison_conditions}' in b);text=text.replace(block,conditions.strip(),1)
 else:
  index=text.index(r'\begin{figure}');text=text[:index]+conditions+'\n\\clearpage\n\n'+text[index:]
 path.write_text(text)
 path=ROOT/'05_thesis/TFM.tex';text=path.read_text();summ=pd.read_csv(A/'all_summary.csv').set_index('model');rows=[]
 for name,r in summ.loc[MAIN_MODELS].sort_values('mean',ascending=False).iterrows():
  row=[LABELS.get(name,SHORT[name]),f'{r["mean"]:.4f}',f'{r.sd:.4f}',f'[{r.low:.4f}, {r.high:.4f}]']
  if name=='BiTFI-TimesFM3':row=[r'\textbf{'+v+'}' for v in row]
  rows.append(row)
 text=replace_tabular(text,'tab:overall',table(['Configuration','NMAE','SD',r'95\% CI'],rows,'lrrr'))
 physical=pd.read_csv(A/'all_physical_MAE.csv').pivot(index='model',columns='variable',values='mean').loc[MAIN_MODELS,cp.COLS];mins=physical.min();rows=[]
 for name,r in physical.iterrows():
  row=[SHORT[name]]
  for v in cp.COLS:
   value=f'{r[v]:.2f}';row.append(r'\textbf{'+value+'}' if r[v]==mins[v] else value)
  rows.append(row)
 headers=['Model',r'$T_{\mathrm{in}}$ ($^\circ$C)',r'$T_{\mathrm{out}}$ ($^\circ$C)','RH',r'CO$_2$ (ppm)',r'Rad (W\,m$^{-2}$)']
 text=replace_tabular(text,'tab:variable',table(headers,rows,'lrrrrr'));path.write_text(text)
 (R/'verification_20260929/table_integration.json').write_text(json.dumps(dict(supplementary_labels=updated+['tab:verified_sensitivity','tab:comparison_conditions'],main_labels=['tab:overall','tab:variable'],author_prose_replaced=False),indent=2))
 print('Updated 2 main and 16 supplementary table bodies; preserved author prose.')
if __name__=='__main__':main()

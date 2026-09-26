"""Refresh standalone figure captions, portable TeX and PDF figure bundles."""
from pathlib import Path
import re,zipfile,json,hashlib
ROOT=Path(__file__).resolve().parents[1];T=ROOT/'05_thesis';F=ROOT/'04_figure'
def balanced(s,start):
 depth=1;i=start
 while depth:
  if s[i]=='{':depth+=1
  elif s[i]=='}':depth-=1
  i+=1
 return s[start:i-1]
md=['# Figure captions'];inventory=[]
for stem in ['TFM','supplementary']:
 s=(T/(stem+'.tex')).read_text();blocks=[]
 for m in re.finditer(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',s,re.S):
  b=m.group();name=re.search(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',b).group(1);i=b.index('\\caption{')+len('\\caption{');caption=balanced(b,i);blocks.append('\\caption{'+caption+'}')
  md.extend(['','## '+name,'',caption]);inventory.append(name)
 (F/(stem+'_captions_20260926.tex')).write_text('\n\n'.join(blocks)+'\n')
(F/'Figure_captions.md').write_text('\n'.join(md)+'\n')
figures=[T/'Figure0_Graphical_abstract.pdf']+[T/n for n in inventory]
assert len(figures)==len(set(figures))
with zipfile.ZipFile(T/'BiTFI_Overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in [T/n for n in ['TFM.tex','supplementary.tex','cas-refs.bib','elsarticle.cls','elsarticle-harv.bst']]+figures:z.write(p,p.name)
with zipfile.ZipFile(F/'AIIA_BiTFI_figures.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in figures:z.write(p,p.name)
 z.write(F/'Figure_captions.md','Figure_captions.md')
 for p in (ROOT/'03_result/revision_experiments_20260926/analysis').glob('*.json'):z.write(p,'validation/'+p.name)
 p=ROOT/'03_result/revision_experiments_20260926/publication_validation.json'
 if p.exists():z.write(p,'validation/publication_validation.json')
 for p in (F/'source_data').glob('*.csv'):z.write(p,'source_data/'+p.name)
 for p in (ROOT/'03_result/revision_experiments_20260926/analysis').glob('*.csv'):
  if p.name not in ['comparison_results.csv','refinement_cases.csv']:z.write(p,'revision_source_data/'+p.name)
# Include validation selection evidence alongside the publication source tables.
v=ROOT/'03_result/refinement_validation_20260926'
if (v/'selection.json').exists():
 with zipfile.ZipFile(F/'AIIA_BiTFI_figures.zip','a',zipfile.ZIP_DEFLATED) as z:
  for name in ['protocol.json','selection.json','summary.csv','greenhouse_scores.csv','manifest_validation.json','a6000_main_test_timing.json','publication_validation.json']:
   p=v/name
   if p.exists():z.write(p,'refinement_validation/'+name)
report=dict(figures=[p.name for p in figures],figure_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in figures},overleaf_archive=str(T/'BiTFI_Overleaf.zip'))
(ROOT/'03_result/revision_experiments_20260926/package_manifest.json').write_text(json.dumps(report,indent=2));print('Packaged',len(figures),'PDF figures')

"""Stage/compile, then explicitly promote a verified hourly publication."""
import argparse,json,re,shutil,subprocess,zipfile,hashlib
from pathlib import Path
import pandas as pd
import pymupdf as fitz
import clean_protocol as cp
R=cp.OUT;P=R/'publication';T=cp.ROOT/'05_thesis';F=cp.ROOT/'04_figure'

def graphical_abstract():
 gain=float(pd.read_csv(R/'analysis/main_paired.csv').set_index('reference').loc['TimesFM3.0-COV-SPA','reduction']);depth=json.loads((R/'refinement_validation/selection.json').read_text())['selected_refinements']
 doc=fitz.open(T/'archive/before_hourly_revision_20260927/Figure0_Graphical_abstract.pdf');page=doc[0]
 spans=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if any(t in s['text'].replace('\xa0',' ') for t in ['lower NMAE','higher NMAE'])];assert len(spans)==1
 s=spans[0];r=fitz.Rect(s['bbox']);strip=fitz.Rect(r.x0,s['origin'][1]-9,r.x1,s['origin'][1]-5);page.add_redact_annot(strip,fill=False,cross_out=False);page.apply_redactions(images=0,graphics=0)
 font='/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf';face=fitz.Font(fontfile=font);page.insert_font(fontname='UpdatedArialBold',fontfile=font)
 label=f'{abs(gain):.2f}% '+('lower' if gain>=0 else 'higher')+' NMAE';size=s['size'];width=face.text_length(label,fontsize=size);rgb=tuple(((s['color']>>shift)&255)/255 for shift in [16,8,0]);page.insert_text(((r.x0+r.x1-width)/2,s['origin'][1]),label,fontname='UpdatedArialBold',fontsize=size,color=rgb)
 heads=[t for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for t in l['spans'] if t['text'].replace('\xa0',' ').startswith('Step 2:')];assert len(heads)==1
 h=heads[0];r=fitz.Rect(h['bbox']);label2=f'Step 2: Covariate refinement ({depth} passes)';page.add_redact_annot(r,fill=False,cross_out=False);page.apply_redactions(images=0,graphics=0);size=min(h['size'],r.width/face.text_length(label2,fontsize=1));width=face.text_length(label2,fontsize=size);rgb=tuple(((h['color']>>shift)&255)/255 for shift in [16,8,0]);page.insert_text(((r.x0+r.x1-width)/2,h['origin'][1]),label2,fontname='UpdatedArialBold',fontsize=size,color=rgb)
 doc.save(P/'Figure0_Graphical_abstract.pdf',garbage=4,deflate=True);doc.close()
 with fitz.open(P/'Figure0_Graphical_abstract.pdf') as check:text=check[0].get_text().replace('\xa0',' ')
 assert label in text and label2 in text

def stage():
 for name in ['cas-refs.bib','elsarticle.cls','elsarticle-harv.bst']:shutil.copy2(T/name,P/name)
 for folder in [R/'figures',R/'figures/Supplementary']:
  for path in folder.glob('Figure*.pdf'):shutil.copy2(path,P/path.name)
 graphical_abstract()

def compile_pdfs():
 for stem in ['TFM','supplementary']:
  log=P/(stem+'_build_output.txt')
  with log.open('w') as out:
   commands=[['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex']]
   if stem=='TFM':commands+=[['bibtex',stem]]
   commands+=[['pdflatex','-interaction=nonstopmode','-halt-on-error',stem+'.tex']]*2
   for cmd in commands:subprocess.run(cmd,cwd=P,stdout=out,stderr=subprocess.STDOUT,check=True)
  text=(P/(stem+'.log')).read_text();assert 'undefined references' not in text.lower() and 'undefined citations' not in text.lower()
  print('Compiled',stem,len(fitz.open(P/(stem+'.pdf'))),'pages',flush=True)

def balanced(text,start):
 depth=1;i=start
 while depth:
  depth+=(text[i]=='{')-(text[i]=='}');i+=1
 return text[start:i-1]

def promote():
 report=R/'publication_validation.json';assert report.exists(),'Run independent publication validation before promotion'
 for p in P.iterdir():
  if p.suffix in ['.tex','.pdf','.bib','.cls','.bst','.docx'] or p.name=='Highlights.txt':shutil.copy2(p,T/p.name)
 F.mkdir(exist_ok=True);inventory=[];captions=['# Figure captions']
 for stem in ['TFM','supplementary']:
  text=(P/(stem+'.tex')).read_text();blocks=[]
  for m in re.finditer(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',text,re.S):
   block=m.group();name=re.search(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',block).group(1);caption=balanced(block,block.index('\\caption{')+len('\\caption{'));captions+=['','## '+name,'',caption];inventory.append(name);blocks.append('\\caption{'+caption+'}')
  (F/(stem+'_captions_20260927.tex')).write_text('\n\n'.join(blocks)+'\n')
 (F/'Figure_captions.md').write_text('\n'.join(captions)+'\n');figures=['Figure0_Graphical_abstract.pdf',*inventory];assert len(set(figures))==len(figures)==17
 for name in figures:shutil.copy2(P/name,F/name)
 archive=F/'archive/source_data_before_hourly_20260927'
 dest=F/'source_data'
 if dest.exists() and not archive.exists():
  archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(dest),str(archive))
 shutil.copytree(R/'figures/source_data',dest,dirs_exist_ok=True)
 with zipfile.ZipFile(T/'BiTFI_Overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in ['TFM.tex','supplementary.tex','cas-refs.bib','elsarticle.cls','elsarticle-harv.bst',*figures]:z.write(P/name,name)
 with zipfile.ZipFile(F/'AIIA_BiTFI_figures.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in figures:z.write(P/name,name)
  z.write(F/'Figure_captions.md','Figure_captions.md');z.write(report,'validation/publication_validation.json')
  for p in dest.rglob('*'):
   if p.is_file():z.write(p,'source_data/'+str(p.relative_to(dest)))
  for folder in ['context_validation','refinement_validation','quality']:
   for name in ['protocol.json','selection.json','selected_contexts.json','summary.csv']:
    p=R/folder/name
    if p.exists():z.write(p,'validation/'+folder+'/'+name)
 hashes={name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in figures};(R/'package_manifest.json').write_text(json.dumps(dict(figures=figures,sha256=hashes,result_root=str(R.relative_to(cp.ROOT))),indent=2));(cp.ROOT/'03_result/active_evaluation.json').write_text(json.dumps(dict(version=R.name,result_root=str(R.relative_to(cp.ROOT)),figures='04_figure',models=21,main_figures=6,supplementary_figures=10,main_comparison_models=8,context_selection=str((R/'context_validation').relative_to(cp.ROOT)),selected_contexts=json.loads((R/'context_validation/selected_contexts.json').read_text()),refinement_selection=str((R/'refinement_validation').relative_to(cp.ROOT)),selected_refinements=json.loads((R/'refinement_validation/selection.json').read_text())['selected_refinements'],validation=str(report.relative_to(cp.ROOT))),indent=2));print('Promoted and packaged 17 figures',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',action='store_true');p.add_argument('--compile',action='store_true');p.add_argument('--promote',action='store_true');a=p.parse_args()
 if a.stage:stage()
 if a.compile:compile_pdfs()
 if a.promote:promote()
if __name__=='__main__':main()

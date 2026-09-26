"""Update only the numerical result in the author-drawn vector abstract."""
from pathlib import Path
import pymupdf as fitz
import pandas as pd
import json
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'03_result/reevaluation_context_20260925'
stats=pd.read_csv(ROOT/'04_figure/source_data/paired_greenhouse_tests.csv').set_index('reference')
gain=float(stats.loc['TimesFM3.0-COV-SPA','reduction_percent'])
doc=fitz.open(ROOT/'05_thesis/Figure0_Graphical_abstract.pdf');page=doc[0]
spans=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if any(x in s['text'].replace('\xa0',' ') for x in ['lower NMAE','higher NMAE'])]
assert len(spans)==1,spans
s=spans[0];rect=fitz.Rect(s['bbox'])
# The source font's tall bounding box overlaps Time and the reference line.
# Redact a narrow strip through the glyphs to preserve those neighboring labels.
strip=fitz.Rect(rect.x0,s['origin'][1]-9,rect.x1,s['origin'][1]-5)
page.add_redact_annot(strip,fill=False,cross_out=False);page.apply_redactions(images=0,graphics=0)
font='/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf';face=fitz.Font(fontfile=font);page.insert_font(fontname='UpdatedArialBold',fontfile=font)
label=f'{abs(gain):.2f}% '+('lower' if gain>=0 else 'higher')+' NMAE';size=s['size'];width=face.text_length(label,fontsize=size)
x=(rect.x0+rect.x1-width)/2;rgb=tuple(((s['color']>>shift)&255)/255 for shift in [16,8,0])
page.insert_text((x,s['origin'][1]),label,fontname='UpdatedArialBold',fontsize=size,color=rgb)
selection=ROOT/'03_result/refinement_validation_20260926/selection.json'
if selection.exists():
 depth=json.loads(selection.read_text())['selected_refinements']
 heads=[t for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for t in l['spans'] if t['text'].replace('\xa0',' ').startswith('Step 2:')]
 assert len(heads)==1,heads
 h=heads[0];r=fitz.Rect(h['bbox']);label2=f'Step 2: Covariate refinement ({depth} passes)'
 page.add_redact_annot(r,fill=False,cross_out=False);page.apply_redactions(images=0,graphics=0)
 fs=min(h['size'],r.width/face.text_length(label2,fontsize=1));w=face.text_length(label2,fontsize=fs)
 rgb2=tuple(((h['color']>>shift)&255)/255 for shift in [16,8,0])
 page.insert_text(((r.x0+r.x1-w)/2,h['origin'][1]),label2,fontname='UpdatedArialBold',fontsize=fs,color=rgb2)
doc.save(ROOT/'04_figure/Figure0_Graphical_abstract.pdf',garbage=4,deflate=True)
print(label)

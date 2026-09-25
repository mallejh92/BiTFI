"""Update only the numerical result in the author-drawn vector abstract."""
from pathlib import Path
import pymupdf as fitz
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'03_result/reevaluation_context_20260925'
stats=pd.read_csv(RUN/'figures/source_data/paired_greenhouse_tests.csv').set_index('reference')
gain=float(stats.loc['TimesFM3.0-COV-SPA','reduction_percent'])
doc=fitz.open(ROOT/'05_thesis/Figure0_Graphical_abstract.pdf');page=doc[0]
spans=[s for b in page.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if 'lower NMAE' in s['text']]
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
doc.save(RUN/'figures/Figure0_Graphical_abstract.pdf',garbage=4,deflate=True)
print(label)

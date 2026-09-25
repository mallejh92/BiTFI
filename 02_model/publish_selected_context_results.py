"""Publish only after evaluation, figure validation and manuscript builds pass."""
from pathlib import Path
import json,shutil,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'03_result/reevaluation_context_20260925';FIG=RUN/'figures';MAN=RUN/'manuscript'
assert (RUN/'validation_report.json').exists()
assert (RUN/'publication_validation.json').exists()
for n in ['TFM.pdf','supplementary.pdf']:assert (MAN/n).exists()
backup=ROOT/'05_thesis/archive/pre_context_reevaluation_20260925';backup.mkdir(exist_ok=True)
for p in (ROOT/'05_thesis').iterdir():
 if p.is_file() and p.suffix in ['.tex','.pdf','.bib','.cls','.bst']:shutil.copy2(p,backup/p.name)
oldfig=ROOT/'04_figure/archive/pre_context_reevaluation_20260925';oldfig.mkdir(parents=True,exist_ok=True)
for p in (ROOT/'04_figure').glob('*.pdf'):shutil.copy2(p,oldfig/p.name)
shutil.copytree(ROOT/'04_figure/Supplementary',oldfig/'Supplementary',dirs_exist_ok=True)
shutil.copytree(ROOT/'04_figure/source_data',oldfig/'source_data',dirs_exist_ok=True)
# Replace source tables as a set to avoid mixing the old S2 diagnostic with selection.
shutil.rmtree(ROOT/'04_figure/source_data');shutil.copytree(FIG/'source_data',ROOT/'04_figure/source_data')
for p in FIG.glob('*.pdf'):shutil.copy2(p,ROOT/'04_figure'/p.name)
for p in (FIG/'Supplementary').glob('*.pdf'):shutil.copy2(p,ROOT/'04_figure/Supplementary'/p.name)
for p in MAN.iterdir():
 if p.suffix in ['.tex','.pdf','.bib','.cls','.bst']:shutil.copy2(p,ROOT/'05_thesis'/p.name)
for name in ['TFM','supplementary']:
 (ROOT/'04_figure'/f'{name}_captions_20260925.tex').write_text('\n'.join(line for line in (MAN/(name+'.tex')).read_text().splitlines() if line.startswith('\\caption'))+'\n')
active=ROOT/'03_result/active_evaluation.json';shutil.copy2(active,backup/'active_evaluation.json')
active.write_text(json.dumps(dict(version='validation-context-20260925',result_root=str(RUN.relative_to(ROOT)),figures='04_figure',models=19,main_figures=6,supplementary_figures=7,context_selection='03_result/context_validation_20260925',selected_contexts=json.loads((ROOT/'03_result/context_validation_20260925/selected_contexts.json').read_text())),indent=2))
# Portable Overleaf package with the same standalone sources and PDF figures.
with zipfile.ZipFile(ROOT/'05_thesis/BiTFI_Overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in MAN.iterdir():
  if p.suffix in ['.tex','.bib','.cls','.bst'] or (p.suffix=='.pdf' and p.name.startswith('Figure')):z.write(p,p.name)
with zipfile.ZipFile(ROOT/'04_figure/AIIA_BiTFI_figures.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in FIG.rglob('*'):
  if p.is_file():z.write(p,p.relative_to(FIG))
print('Published',RUN)

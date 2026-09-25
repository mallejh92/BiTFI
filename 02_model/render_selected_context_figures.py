"""Render reviewed manuscript figures into the isolated reevaluation directory."""
import os,sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'03_result/reevaluation_context_20260925'
os.environ['BITFI_CLEAN']='1';os.environ['BITFI_RESULT_ROOT']=str(RUN)
sys.path[:0]=[str(ROOT/'02_model/figures')]
import pandas as pd,numpy as np
import build_prism_figures as b
from FigureS7_univariate_backbones import main as backbone
from FigureS8_moment_tuning import main as moment
b.OUT.mkdir(exist_ok=True);b.DATA.mkdir(exist_ok=True)
# Unchanged dataset and diagram, including user's own graphical abstract.
for name in ['Figure0_Graphical_abstract.pdf','Figure1_Dataset_profile.pdf','Figure2_BiTFI_framework.pdf','Figure2_BiTFI_caption.md']:
 p=ROOT/'04_figure'/name
 if not p.exists():p=ROOT/'05_thesis'/name
 shutil.copy2(p,b.OUT/name)
shutil.copytree(ROOT/'04_figure/source_data/Figure1_profile',b.DATA/'Figure1_profile',dirs_exist_ok=True)
b.ps.setup();full,main=b.load_results()
for name in full.model.unique():
 if name.startswith('CAFI'):continue
 p=RUN/'evaluation'/name;assert (p/'complete.json').exists(),name
 d=full[full.model==name];assert d.case_id.nunique()==3357 and np.isfinite(d.NMAE).all(),name
b.overall(full);b.gap_robustness(full);b.variables(full);b.examples();b.supplementary(full,main);backbone();moment()
full.groupby('model').agg(rows=('NMAE','size'),source=('source_directory','first'),context_len=('context_len','first')).to_csv(b.DATA/'result_sources.csv')
print('All figures rendered',b.OUT)

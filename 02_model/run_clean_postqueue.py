"""Additional jobs to execute after a GPU's main evaluation queue completes."""
import subprocess,sys,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'03_result/reevaluation_clean_20260911';family=sys.argv[1];env={**os.environ,'BITFI_CLEAN':'1'}
expected=['TimesFM3.0-COV-SPA'] if family=='tfm3' else ['CAFI'] if family=='chronos' else ['MOMENT-FT']
while not all((out/'evaluation'/m/'complete.json').exists() for m in expected):time.sleep(20)
jobs=[]
if family=='tfm3':
 jobs=[['run_univariate_backbone_comparison.py','--model','TimesFM3.0'],['export_clean_figure6.py','--model','BiTFI-TimesFM3'],['run_clean_auxiliary.py','--family','tfm3']]
elif family=='chronos':
 jobs=[['run_univariate_backbone_comparison.py','--model','Chronos2'],['run_univariate_backbone_comparison.py','--model','TimesFM2.5'],['run_clean_auxiliary.py','--family','chronos']]
else:
 jobs=[['export_clean_figure6.py','--model',m] for m in ['SAITS','MOMENT-FT','Spatial-Ridge']]
for j in jobs:
 if j[0]=='run_univariate_backbone_comparison.py' and (out/'univariate_backbone_comparison'/f'{j[-1]}_complete.json').exists():continue
 if j[0]=='export_clean_figure6.py' and (out/'figure6_common_window'/f"{'BiTFI' if j[-1]=='BiTFI-TimesFM3' else j[-1]}.csv").exists():continue
 if j[0]=='run_clean_auxiliary.py' and (out/f'{family}_auxiliary_complete.json').exists():continue
 cmd=[sys.executable,'-u',str(ROOT/'02_model'/j[0]),*j[1:]]
 print('START',j,flush=True);subprocess.run(cmd,env=env,cwd=ROOT,check=True)
print('POSTQUEUE COMPLETE',family,flush=True)

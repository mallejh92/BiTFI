"""Dispatch independent configurations after a GPU's earlier job completes."""
from pathlib import Path
import subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'03_result/reevaluation_context_20260925'
while not (out/'evaluation'/sys.argv[1]/'complete.json').exists():time.sleep(10)
subprocess.run([sys.executable,'-u',str(ROOT/'02_model/run_selected_context_queue.py'),*sys.argv[2:]],check=True,cwd=ROOT)

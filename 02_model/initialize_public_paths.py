"""Resolve the released split filenames for a fresh checkout without overwriting runs."""
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args()
 source=json.loads((ROOT/'01_data/split.json').read_text());out={}
 for group in ['train','test']:
  out[group]=[]
  for name in source[group]:
   hits=list((ROOT/'01_data').rglob(Path(name).name))
   if len(hits)!=1:raise ValueError(f'{name}: expected one matching input file, got {len(hits)}')
   out[group].append(str(hits[0].resolve()))
 out['meta']=source.get('meta',{})
 target=ROOT/'03_result/comparison/split.json'
 if a.check_only:print('Resolved',len(out['train']),'training candidates and',len(out['test']),'test sites; no files written');return
 if target.exists():raise FileExistsError(f'{target} already exists; leave the existing experiment intact')
 target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(out,indent=2));print(target)
if __name__=='__main__':main()

"""Populate the project cache with the exact released checkpoint revisions.

Weights stay outside the source release. Run once with network access, then use
HF_HUB_OFFLINE=1 to prevent a moving repository main branch changing the run.
"""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args();cache=ROOT/'03_result/model_cache/huggingface'
 for m in json.loads((ROOT/'02_model/environments/checkpoints.json').read_text()):
  directory=cache/m['cache_subdirectory'];repo=directory/('models--'+m['repo_id'].replace('/','--'));snapshot=repo/'snapshots'/m['revision']
  if not a.check_only:
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id=m['repo_id'],revision=m['revision'],allow_patterns=m['files'],cache_dir=directory)
   refs=repo/'refs';refs.mkdir(exist_ok=True);(refs/'main').write_text(m['revision'])
  for file in m['files']:
   assert (snapshot/file).is_file(),f'Missing {m["repo_id"]}: {file}; run without --check-only to download'
  if m['cache_subdirectory']=='hub':assert (repo/'refs/main').read_text().strip()==m['revision']
  print(m['repo_id'],m['revision'],'ready')
if __name__=='__main__':main()

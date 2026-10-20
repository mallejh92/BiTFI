"""Sequential single-GPU queue for the revision-2 BiTFI variants (E2).

One job at a time on the given GPU (1 or 2 only). Jobs are 'repro', 'smoke:<variant>'
or 'shard<i>:<variant>'. A shard job waits for that variant's passed smoke check.
After a shard finishes, both shards present -> merge into evaluation/<variant>
(standard environment, hourly_analysis.save_merged checks).
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ['BITFI_RUN_ROOT']).resolve()
OUT = RUN / 'revision/bitfi_variants'
LOG = OUT / 'logs'
SCRIPT = ROOT / '02_model/revision2_bitfi_variants.py'


def environment(gpu):
    env = os.environ.copy()
    env.update(BITFI_RUN_ROOT=str(RUN), BITFI_CLEAN='1', JAX_PLATFORMS='cpu',
               HF_HOME=str(ROOT / '03_result/model_cache/huggingface'), HF_HUB_OFFLINE='1',
               CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    return env


def execute(name, command, env):
    done = LOG / f'{name}.done.json'
    if done.exists():
        return True
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'START', name, flush=True)
    start = time.time()
    with (LOG / f'{name}.log').open('a') as log:
        result = subprocess.run([str(c) for c in command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    status = dict(task=name, command=[str(c) for c in command], gpu=env['CUDA_VISIBLE_DEVICES'],
                  returncode=result.returncode, elapsed_s=time.time() - start)
    if result.returncode:
        (LOG / f'{name}.failed.json').write_text(json.dumps(status, indent=2))
        print(time.strftime('%Y-%m-%d %H:%M:%S'), 'FAILED', name, flush=True)
        return False
    done.write_text(json.dumps(status, indent=2))
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'DONE', name, round(status['elapsed_s']), 's', flush=True)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True, choices=[1, 2])
    p.add_argument('--jobs', required=True)
    a = p.parse_args()
    LOG.mkdir(parents=True, exist_ok=True)
    env = environment(a.gpu)
    for job in a.jobs.split(','):
        kind, _, variant = job.partition(':')
        if kind == 'repro':
            ok = execute('reproduction', [ROOT / '.venv_tfm3/bin/python', '-u', SCRIPT, '--repro'], env)
        elif kind == 'smoke':
            ok = execute(f'{variant}_smoke', [ROOT / '.venv_tfm3/bin/python', '-u', SCRIPT, '--variant', variant, '--smoke'], env)
        elif kind.startswith('shard'):
            shard = int(kind[5:])
            while not (LOG / f'{variant}_smoke.done.json').exists():
                if (LOG / f'{variant}_smoke.failed.json').exists():
                    raise SystemExit(f'smoke failed for {variant}')
                time.sleep(30)
            ok = execute(f'{variant}_shard{shard}', [ROOT / '.venv_tfm3/bin/python', '-u', SCRIPT, '--variant', variant,
                                                     '--shard', shard, '--shards', 2], env)
            if ok and all((OUT / variant / f'shard{i}/complete.json').exists() for i in range(2)):
                merged = env | dict(CUDA_VISIBLE_DEVICES='')
                ok = execute(f'{variant}_merge', [ROOT / '.venv/bin/python', '-u', SCRIPT, '--variant', variant, '--merge'], merged)
        else:
            raise SystemExit(f'unknown job {job}')
        if not ok:
            raise SystemExit(f'job failed: {job}')
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'QUEUE COMPLETE', a.jobs, flush=True)


if __name__ == '__main__':
    main()

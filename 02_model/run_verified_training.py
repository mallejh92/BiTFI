"""Isolated validation-only budget checks for existing SAITS/MOMENT baselines.

Run with CUDA_VISIBLE_DEVICES=2 and BITFI_RUN_ROOT pointing to the verified copy.
No reads of test errors occur during the training/selection stage.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get('BITFI_RUN_ROOT', '')).resolve()
EXPECTED = ROOT / '03_result/revision_verified_20260929'
if RUN != EXPECTED:
    raise RuntimeError('This budget-check driver only writes to the verified revision copy')
AUDIT = RUN / 'verified_training'
STAMP = 'before_verified_20260929'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(name, value):
    (AUDIT / name).write_text(json.dumps(value, indent=2))


def run_logged(command, filename):
    started = time.time()
    print('RUN', command, flush=True)
    with (AUDIT / filename).open('w') as log:
        completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f'{filename} failed with status {completed.returncode}')
    return time.time() - started


def train():
    AUDIT.mkdir(exist_ok=True)
    plan_path = AUDIT / 'training_plan.json'
    if plan_path.exists():
        raise RuntimeError('A recorded budget-check plan already exists; inspect it before another training stage')
    model_specs = {
        'MOMENT': {'learning_rates': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2], 'max_epochs': 300, 'patience': 20,
                   'rationale': 'Retain all prior rates and extend two log-spaced rates above the former optimum; longer patience accommodates noisy slow head adaptation.'},
        'SAITS': {'learning_rate': .001, 'max_epochs': 300, 'patience': 10,
                  'rationale': 'Retain architecture, learning rate, seed, corruption and patience; extend only the epoch cap beyond the former best epoch 60.'}}
    plan = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_run': str(ROOT / '03_result/reevaluation_hourly_20260927'),
            'run_root': str(RUN), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'seed': 42,
            'selection': 'Minimum existing fixed validation masked MAE; training choices fixed before any revised test evaluation.',
            'no_test_selection': True, 'main_matched_saits_unchanged': True, 'models': model_specs,
            'input_sha256': {str(p.relative_to(RUN)): sha(p) for p in [RUN / 'models/SAITS/windows.npz', RUN / 'models/MOMENT/frozen_features.pt', RUN / 'split.json']}}
    for model in model_specs:
        folder = RUN / 'models' / model
        backup = folder.with_name(model + '_' + STAMP)
        if backup.exists():
            raise RuntimeError(f'Backup already exists: {backup}')
        plan['models'][model]['baseline_complete'] = json.loads((folder / 'complete.json').read_text())
        plan['models'][model]['baseline_checkpoint_sha256'] = sha(folder / 'best.pt')
    write('training_plan.json', plan)
    for model in model_specs:
        folder = RUN / 'models' / model
        backup = folder.with_name(model + '_' + STAMP)
        folder.rename(backup)
        folder.mkdir()
        retained = ['windows.npz', 'training_protocol.json'] if model == 'SAITS' else ['frozen_features.pt']
        for name in retained:
            shutil.copy2(backup / name, folder / name)
    elapsed = {}
    elapsed['MOMENT'] = run_logged([sys.executable, '02_model/train_moment_head.py', '--epochs', '300', '--patience', '20', '--learning-rates', '0.0001', '0.0003', '0.001', '0.003', '0.01'], 'MOMENT_training.log')
    elapsed['SAITS'] = run_logged([sys.executable, '02_model/train_saits.py', '--clean', '--out', str(RUN / 'models/SAITS'), '--epochs', '300', '--patience', '10'], 'SAITS_training.log')
    diagnostics = {'created_utc': datetime.now(timezone.utc).isoformat(), 'wall_seconds': elapsed, 'models': {}}
    for model in model_specs:
        folder = RUN / 'models' / model
        new = json.loads((folder / 'complete.json').read_text())
        old = plan['models'][model]['baseline_complete']
        h = pd.read_csv(folder / 'history.csv')
        old_h = pd.read_csv(folder.with_name(model + '_' + STAMP) / 'history.csv')
        join = ['epoch'] if model == 'SAITS' else ['learning_rate', 'epoch']
        pairs = old_h.merge(h, on=join, suffixes=('_old', '_new'), validate='one_to_one')
        diagnostics['models'][model] = {'training_complete': new, 'old_validation_mae': old['best_validation_mae'],
            'new_validation_mae': new['best_validation_mae'], 'validation_improvement': old['best_validation_mae'] - new['best_validation_mae'],
            'prefix_comparison_rows': len(pairs), 'prefix_max_abs_validation_delta': float((pairs.validation_mae_old - pairs.validation_mae_new).abs().max()),
            'checkpoint_sha256': sha(folder / 'best.pt')}
        if new['best_validation_mae'] > old['best_validation_mae']:
            raise RuntimeError(f'{model} did not improve validation; inspect before selecting its checkpoint')
    write('validation_diagnostics.json', diagnostics)
    print(json.dumps(diagnostics, indent=2), flush=True)



def extend_moment():
    folder = RUN / 'models/MOMENT'
    before = json.loads((folder / 'complete.json').read_text())
    if all(r['stopped_early'] for r in before['runs']):
        raise RuntimeError('Every candidate already stopped early; no cap extension is justified')
    archive = AUDIT / 'MOMENT_stage1_cap300'
    archive.mkdir()
    for filename in ['best.pt', 'history.csv', 'search.csv', 'training_protocol.json', 'complete.json']:
        shutil.copy2(folder / filename, archive / filename)
    shutil.copy2(AUDIT / 'validation_diagnostics.json', archive / 'validation_diagnostics.json')
    amendment = {'created_utc': datetime.now(timezone.utc).isoformat(), 'reason': 'The 1e-4 candidate exhausted 300 epochs while still improving; inspect convergence without using test results.',
                 'max_epochs': 1000, 'patience': 20, 'learning_rates': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                 'other_training_conditions_unchanged': True, 'test_evaluation_started': False}
    write('MOMENT_budget_amendment.json', amendment)
    seconds = run_logged([sys.executable, '02_model/train_moment_head.py', '--epochs', '1000', '--patience', '20', '--learning-rates', '0.0001', '0.0003', '0.001', '0.003', '0.01'], 'MOMENT_training_extended.log')
    after = json.loads((folder / 'complete.json').read_text())
    if after['best_validation_mae'] > before['best_validation_mae']:
        raise RuntimeError('Extended search lost the previous validation minimum')
    previous = pd.read_csv(archive / 'history.csv')
    current = pd.read_csv(folder / 'history.csv')
    pairs = previous.merge(current, on=['learning_rate', 'epoch'], suffixes=('_old', '_new'), validate='one_to_one')
    delta = float((pairs.validation_mae_old - pairs.validation_mae_new).abs().max())
    diagnostics = json.loads((AUDIT / 'validation_diagnostics.json').read_text())
    d = diagnostics['models']['MOMENT']
    d.update(training_complete=after, new_validation_mae=after['best_validation_mae'],
             validation_improvement=d['old_validation_mae'] - after['best_validation_mae'],
             checkpoint_sha256=sha(folder / 'best.pt'), stage1_prefix_max_abs_validation_delta=delta,
             all_candidates_stopped_early=all(r['stopped_early'] for r in after['runs']))
    diagnostics['wall_seconds']['MOMENT_extension'] = seconds
    write('validation_diagnostics.json', diagnostics)
    print(json.dumps(d, indent=2), flush=True)



def freeze_selection():
    import torch
    if (AUDIT / 'selection_frozen.json').exists():
        raise RuntimeError('Selection has already been frozen; do not replace it after evaluation')
    diagnostics = json.loads((AUDIT / 'validation_diagnostics.json').read_text())
    plan = json.loads((AUDIT / 'training_plan.json').read_text())
    if not diagnostics['models']['MOMENT'].get('all_candidates_stopped_early') or not diagnostics['models']['SAITS']['training_complete']['stopped_early']:
        raise RuntimeError('Inspect incomplete convergence diagnostics before freezing selection')
    record = {'frozen_utc': datetime.now(timezone.utc).isoformat(), 'selection': 'Minimum fixed training-site validation masked MAE; no revised test results used', 'models': {}}
    for model in ['MOMENT', 'SAITS']:
        checkpoint = RUN / 'models' / model / 'best.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        digest = sha(checkpoint)
        assert digest == diagnostics['models'][model]['checkpoint_sha256']
        source = Path(plan['source_run']) / 'models' / model / 'best.pt'
        assert sha(source) == plan['models'][model]['baseline_checkpoint_sha256']
        record['models'][model] = {'epoch': state['epoch'], 'validation_mae': state['validation_mae'], 'learning_rate': state.get('learning_rate', .001), 'checkpoint_sha256': digest, 'seed': state['seed']}
    write('selection_frozen.json', record)
    print(json.dumps(record, indent=2), flush=True)


def evaluate(models):
    if not (AUDIT / 'selection_frozen.json').exists():
        raise RuntimeError('Validation diagnostics must be reviewed and selection frozen before evaluation')
    reports = {}
    for model in models:
        destination = RUN / 'evaluation' / model
        backup = destination.with_name(model + '_' + STAMP)
        if destination.exists() and not backup.exists():
            destination.rename(backup)
        elif destination.exists():
            raise RuntimeError(f'New evaluation exists for {model}; inspect before resuming')
        smoke = RUN / 'smoke' / model
        if smoke.exists():
            smoke.rename(smoke.with_name(model + '_' + STAMP))
        smoke_seconds = run_logged([sys.executable, '02_model/run_clean_evaluation.py', '--models', model, '--smoke'], model + '_smoke.log')
        full_seconds = run_logged([sys.executable, '02_model/run_clean_evaluation.py', '--models', model], model + '_evaluation.log')
        d = pd.read_csv(destination / 'results.csv').query("group_type=='all'")
        manifest = pd.read_csv(RUN / 'mask_manifest.csv')
        expected = {(int(r.case_id), v) for r in manifest.itertuples() for v in r.masked_vars.split(',')}
        if set(zip(d.case_id, d.variable)) != expected or d.duplicated(['case_id', 'variable']).any() or len(d) != 6205:
            raise AssertionError(f'Incomplete or duplicate evaluation for {model}')
        if not np.isfinite(d[['MAE', 'NMAE']]).all().all():
            raise AssertionError(f'Nonfinite metrics for {model}')
        paths = [destination / 'predictions' / f'{case_id}.npy' for case_id in manifest.case_id]
        if not all(p.exists() for p in paths):
            raise AssertionError(f'Incomplete prediction arrays for {model}')
        g = d.assign(weighted=d.NMAE * d.n_eval).groupby('greenhouse')[['weighted', 'n_eval']].sum()
        reports[model] = {'cases': len(manifest), 'variable_cases': len(d), 'sites': len(g), 'NMAE': float((g.weighted / g.n_eval).mean()),
                          'smoke_wall_seconds': smoke_seconds, 'evaluation_wall_seconds': full_seconds, 'predictions': len(paths),
                          'results_sha256': sha(destination / 'results.csv')}
        write('evaluation_' + model + '.json', reports[model])
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['train', 'extend-moment', 'freeze-selection', 'evaluate'])
    parser.add_argument('--models', nargs='+', choices=['SAITS', 'MOMENT-FT'], default=['SAITS', 'MOMENT-FT'])
    args = parser.parse_args()
    if os.environ.get('BITFI_CLEAN') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise RuntimeError('Set BITFI_CLEAN=1 and CUDA_VISIBLE_DEVICES=2 for this isolated run')
    if args.stage == 'train':train()
    elif args.stage == 'extend-moment':extend_moment()
    elif args.stage == 'freeze-selection':freeze_selection()
    else:evaluate(args.models)

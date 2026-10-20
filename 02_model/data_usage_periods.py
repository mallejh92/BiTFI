"""Explicit recording intervals applied before splitting, scaling or masking."""
import json
from pathlib import Path
import pandas as pd


def load_usage_periods(run_root):
    path = Path(run_root) / 'data_usage_periods.json'
    if not path.exists():
        return {}
    sites = json.loads(path.read_text())['sites']
    for name, rule in sites.items():
        if not isinstance(name, str) or not isinstance(rule, dict):
            raise ValueError('Each usage period must name a site and its interval')
        bounds = []
        for key in ['start', 'end']:
            value = rule.get(key)
            if value is None:
                bounds.append(None)
                continue
            stamp = pd.Timestamp(value)
            if pd.isna(stamp) or stamp.tz is not None or stamp != stamp.floor('h'):
                raise ValueError(f'{name}: {key} must be a local whole-hour timestamp')
            bounds.append(stamp)
        if all(value is None for value in bounds):
            raise ValueError(f'{name}: no recording bound specified')
        if all(value is not None for value in bounds) and bounds[0] > bounds[1]:
            raise ValueError(f'{name}: start is after end')
    return sites


def restrict_record(frame, rule):
    """Keep an inclusive calendar interval without joining separated observations."""
    original = dict(original_source_rows=len(frame),
                    original_start=str(frame.index[0]),
                    original_end=str(frame.index[-1]))
    retained = frame.loc[rule.get('start'):rule.get('end')].copy()
    if retained.empty:
        raise ValueError(f'Usage interval contains no source rows: {rule}')
    retained.attrs.update(original)
    retained.attrs['usage_period'] = dict(rule)
    return retained

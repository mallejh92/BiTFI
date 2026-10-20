"""Recording bounds precede regularization, splitting and scaler fitting."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_usage_periods import load_usage_periods
from preprocessing import _load_file


class RecordingPeriods(unittest.TestCase):
    def test_cutoff_preserves_hours_and_excludes_later_extrema(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'sensor.csv'
            pd.DataFrame({'date': ['2025-01-01 00:00', '2025-01-01 02:00',
                                   '2025-01-01 03:00', '2025-02-01 00:00'],
                          'value': [1., 3., 4., 999.]}).to_csv(source, index=False)
            d = _load_file(source, usage_period={'end': '2025-01-01 03:00:00'})
            self.assertEqual(len(d), 4)
            self.assertTrue(np.isnan(d.iloc[1, 0]))
            self.assertEqual(d.iloc[-1, 0], 4.)
            self.assertEqual(d.attrs['source_rows'], 3)
            self.assertEqual(d.attrs['inserted_hourly_rows'], 1)
            self.assertEqual(d.attrs['original_source_rows'], 4)
            scaler = MinMaxScaler().fit(d.iloc[:int(.8 * len(d))].dropna())
            self.assertEqual(scaler.data_max_[0], 3.)

    def test_policy_rejects_invalid_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for rule in [{'end': '2025-01-01 00:30'}, {},
                         {'start': '2025-02-01', 'end': '2025-01-01'}]:
                (root / 'data_usage_periods.json').write_text(json.dumps({'sites': {'x': rule}}))
                with self.assertRaises(ValueError):
                    load_usage_periods(root)

    def test_absent_policy_retains_original_protocol(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(load_usage_periods(folder), {})


if __name__ == '__main__':
    unittest.main()

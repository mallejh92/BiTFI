"""Regression checks for elapsed-time preservation (no model inference)."""
import unittest
import numpy as np
import pandas as pd
from preprocessing import regularize_hourly,assert_hourly
class HourlyGrid(unittest.TestCase):
 def test_missing_hour_remains_missing(self):
  original=pd.DataFrame({'x':[2.,7.]},index=pd.to_datetime(['2025-01-01 00:00','2025-01-01 02:00']))
  actual=regularize_hourly(original)
  self.assertEqual(len(actual),3);self.assertTrue(np.isnan(actual.iloc[1,0]));self.assertEqual(actual.iloc[2,0],7.)
  pd.testing.assert_frame_equal(actual.loc[original.index],original,check_freq=False)
 def test_duplicate_and_off_hour_fail(self):
  for dates in [['2025-01-01','2025-01-01'],['2025-01-01 00:30','2025-01-01 01:30']]:
   with self.assertRaises(ValueError):regularize_hourly(pd.DataFrame({'x':[1,2]},index=pd.to_datetime(dates)))
 def test_daily_reference_is_24_hours(self):
  index=pd.date_range('2025-01-01',periods=50,freq='h').delete([5,9,11,27])
  d=regularize_hourly(pd.DataFrame({'x':index.hour.to_numpy(float)},index=index))
  self.assertEqual(d.index[48]-d.index[24],pd.Timedelta(hours=24));self.assertEqual(d.iloc[48,0],d.iloc[24,0])
 def test_irregular_index_rejected(self):
  with self.assertRaises(AssertionError):assert_hourly(pd.to_datetime(['2025-01-01','2025-01-03']))
class NoPastContext(unittest.TestCase):
 def test_leading_missing_block_never_calls_forecaster(self):
  from models.foundation_model import TimesFM3MVImputation
  from unittest.mock import Mock
  model=object.__new__(TimesFM3MVImputation);model.tfm=object()
  model._predict_target=Mock(side_effect=AssertionError('No past observations'))
  index=pd.date_range('2025-01-01',periods=6,freq='h')
  masked=pd.DataFrame({'Tin':[np.nan,np.nan,np.nan,np.nan,8.,9.]},index=index)
  valid=masked.notna().astype(float);artificial=pd.DataFrame({'Tin':[False,False,True,True,False,False]},index=index)
  base=masked.interpolate(limit_direction='both')
  restored=model._impute_variant(masked,valid,base,artificial)
  model._predict_target.assert_not_called()
  np.testing.assert_array_equal(restored.Tin,[8.,8.,8.,8.,8.,9.])
if __name__=='__main__':unittest.main()

"""Test 1D detectors on some simple cases.

We only test detectors with return_list=False, because return_list=True is
effectively tested by test_list_label_convert.py
"""
from math import isnan

import numpy as np
import pandas as pd
import pytest
from packaging.version import parse
from sklearn.svm import SVR

import adtk.detector as detector
from adtk._base import _TrainableModel
from adtk._utils import PandasBugError

nan = float("nan")

testCases = [
    {
        "model": detector.ThresholdAD,
        "params": {},
        "s": [0, -10, 0, nan, 0, 10],
        "a": [0, 0, 0, nan, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.ThresholdAD,
        "params": {"low": -5},
        "s": [0, -10, 0, nan, 0, 10],
        "a": [0, 1, 0, nan, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.ThresholdAD,
        "params": {"low": -5, "high": 5},
        "s": [0, -10, 0, nan, 0, 10],
        "a": [0, 1, 0, nan, 0, 1],
        "pandas_bug": False,
    },
    {
        "model": detector.QuantileAD,
        "params": {},
        "s": [0, -10, 0, nan, 0, 10],
        "a": [0, 0, 0, nan, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.QuantileAD,
        "params": {"low": 0.1},
        "s": [0, -10, 0, nan, 0, 10],
        "a": [0, 1, 0, nan, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.QuantileAD,
        "params": {"low": 0.1, "high": 0.9},
        "s": [0, -10, 0, nan, 0, 10],
        "a": [0, 1, 0, nan, 0, 1],
        "pandas_bug": False,
    },
    {
        "model": detector.InterQuartileRangeAD,
        "params": {},
        "s": [0, -10, 0, 0, 0, nan, 0, 10],
        "a": [0, 1, 0, 0, 0, nan, 0, 1],
        "pandas_bug": False,
    },
    {
        "model": detector.InterQuartileRangeAD,
        "params": {"c": (None, 3)},
        "s": [0, -10, 0, 0, 0, nan, 0, 10],
        "a": [0, 0, 0, 0, 0, nan, 0, 1],
        "pandas_bug": False,
    },
    {
        "model": detector.InterQuartileRangeAD,
        "params": {"c": (3, None)},
        "s": [0, -10, 0, 0, 0, nan, 0, 10],
        "a": [0, 1, 0, 0, 0, nan, 0, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.GeneralizedESDTestAD,
        "params": {},
        "s": [0, -10, 0, 0, 0, nan, 0, 10] + [0] * 10,
        "a": [0, 1, 0, 0, 0, nan, 0, 1] + [0] * 10,
        "pandas_bug": False,
    },
    {
        "model": detector.GeneralizedESDTestAD,
        "params": {"alpha": 0.0001},
        "s": [0, -10, 0, 0, 0, nan, 0, 10] + [0] * 10,
        "a": [0, 0, 0, 0, 0, nan, 0, 0] + [0] * 10,
        "pandas_bug": False,
    },
    {
        "model": detector.PersistAD,
        "params": {},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0],
        "a": [nan, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, nan, nan, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.PersistAD,
        "params": {"side": "positive"},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0],
        "a": [nan, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, nan, nan, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.PersistAD,
        "params": {"side": "negative"},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0],
        "a": [nan, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, nan, nan, 0],
        "pandas_bug": False,
    },
    {
        "model": detector.PersistAD,
        "params": {"window": 2},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0] + [1, 0] * 8,
        "a": [nan, nan, 0, 0, 0, 1, 1, 0, 0, 0, 1, nan, nan, nan] + [0, 0] * 8,
        "pandas_bug": False,
    },
    {
        "model": detector.PersistAD,
        "params": {"window": "50H", "min_periods": 2},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0] + [1, 0] * 8,
        "a": [nan, nan, 0, 0, 0, 1, 1, 0, 0, 0, 1, nan, nan, nan] + [0, 0] * 8,
        "pandas_bug": True,
    },
    {
        "model": detector.PersistAD,
        "params": {"window": 2, "min_periods": 1},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0] + [1, 0] * 8,
        "a": [nan, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, nan, 0, 1] + [0, 0] * 8,
        "pandas_bug": False,
    },
    {
        "model": detector.PersistAD,
        "params": {"window": "50H", "min_periods": 1},
        "s": [0, 1, 0, 1, 0, -10, -9, -10, -9, -10, 1, nan, 1, 0] + [1, 0] * 8,
        "a": [nan, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, nan, 0, 1] + [0, 0] * 8,
        "pandas_bug": True,
    },
    {
        "model": detector.PersistAD,
        "params": {"c": 1.5},
        "s": [0, 1, 2.1, 3.3, 4.5, 5.8, 7.3],
        "a": [nan, 0, 0, 0, 0, 0, 1],
        "pandas_bug": False,
    },
    {
        "model": detector.LevelShiftAD,
        "params": {"window": 3},
        "s": [0] * 10 + [1] * 10,
        "a": [nan] * 3 + [0] * 6 + [1] * 3 + [0] * 6 + [nan] * 2,
        "pandas_bug": False,
    },

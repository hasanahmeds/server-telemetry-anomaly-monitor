"""Test 1D transformors on some simple cases."""
import sys

import pandas as pd
import pytest
from packaging.version import parse

import adtk.transformer as transformer
from adtk._base import _TrainableModel
from adtk._utils import PandasBugError

nan = float("nan")

testCases = [
    {
        "model": transformer.StandardScale,
        "params": {},
        "s": [nan, 0, 1, nan, 2, nan],
        "t": [nan, -1, 0, nan, 1, nan],
        "pandas_bug": False,
    },
    {
        "model": transformer.StandardScale,
        "params": {},
        "s": [nan, 1, 1, nan, 1, nan],
        "t": [nan, 0, 0, nan, 0, nan],
        "pandas_bug": False,
    },
    {
        "model": transformer.RollingAggregate,
        "params": {"agg": "mean", "window": 3, "center": True},
        "s": [0, 1, 2, 3, nan, 5, 6, 7, 8, 9],
        "t": [nan, 1, 2, nan, nan, nan, 6, 7, 8, nan],
        "pandas_bug": False,
    },
    {
        "model": transformer.RollingAggregate,
        "params": {"agg": "mean", "window": 3, "center": False},
        "s": [0, 1, 2, 3, nan, 5, 6, 7, 8, 9],
        "t": [nan, nan, 1, 2, nan, nan, nan, 6, 7, 8],

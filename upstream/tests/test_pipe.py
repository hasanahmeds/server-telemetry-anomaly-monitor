"""Test pipeline and pipenet.

We do not test graph plot because it is covered by test_visualization."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import adtk.aggregator as aggregator
import adtk.detector as detector
import adtk.transformer as transformer
from adtk.pipe import Pipeline, Pipenet


def test_pipenet_default():
    """
    Test default setting of pipenet
    """
    df = pd.DataFrame(
        np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 10, 20, 30, 41, 50, 60, 70, 80, 90],
                [0, 100, 200, 300, 400, 500, 601, 700, 800, 900],
                [0, 0, 0, 0, 0, 0, 0, 100, 0, 0],
            ]
        ).T,
        index=pd.date_range(start="2017-1-1", periods=10, freq="D"),
        columns=["A", "B", "C", "D"],
    )

    reg_ab = LinearRegression()
    reg_ac = LinearRegression()
    my_pipe = Pipenet(
        {
            "A-B-regression-ad": {
                "model": detector.RegressionAD(regressor=reg_ab, target="B"),
                "input": "original",
                "subset": ["A", "B"],
            },
            "A-C-regression-error": {
                "model": transformer.RegressionResidual(
                    regressor=reg_ac, target="C"
                ),
                "input": "original",
                "subset": ["A", "C"],
            },
            "A-C-regression-ad": {
                "model": detector.InterQuartileRangeAD(),
                "input": "A-C-regression-error",
                "subset": "all",
            },
            "ABC-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["A-B-regression-ad", "A-C-regression-ad"],
            },
            "D-ad": {
                "model": detector.QuantileAD(high=0.9, low=0.1),
                "input": "original",
                "subset": ["D"],
            },
            "ABCD-ad": {
                "model": aggregator.OrAggregator(),
                "input": ["ABC-ad", "D-ad"],
            },
        }
    )

    anomaly = my_pipe.fit_detect(df)
    pd.testing.assert_series_equal(
        anomaly,
        pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 0], index=df.index),
        check_dtype=False,
    )

    assert (
        my_pipe.score(
            df,
            pd.Series([0, 0, 0, 0, 1, 0, 1, 1, 0, 0], index=df.index),

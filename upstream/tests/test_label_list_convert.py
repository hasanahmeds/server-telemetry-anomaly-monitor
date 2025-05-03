"""Test converter between binary labels and event lists
"""

import numpy as np
import pandas as pd
from pandas import Timestamp

from adtk.data import to_events, to_labels


def test_binary_label_to_list_freq_as_period_merge_consecutive():
    binary_series = pd.DataFrame(
        np.array(
            [
                [0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ).T,
        index=pd.date_range(start=0, periods=10, freq="1d"),
        columns=["type1", "type2", "type3", "type4"],
    )

    anomaly_list = to_events(
        binary_series, freq_as_period=True, merge_consecutive=True
    )

    anomaly_list_true = {
        "type1": [
            (
                Timestamp("1970-01-02 00:00:00"),
                Timestamp("1970-01-02 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-05 00:00:00"),
                Timestamp("1970-01-06 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-09 23:59:59.999999999"),
            ),
        ],
        "type2": [
            (
                Timestamp("1970-01-01 00:00:00"),
                Timestamp("1970-01-01 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-10 23:59:59.999999999"),
            ),
        ],
        "type3": [],
        "type4": [
            (
                Timestamp("1970-01-01 00:00:00"),
                Timestamp("1970-01-10 23:59:59.999999999"),
            )
        ],
    }

    assert anomaly_list == anomaly_list_true

    for i in range(4):
        assert (
            to_events(binary_series["type{}".format(i + 1)])
            == anomaly_list_true["type{}".format(i + 1)]
        )


def test_binary_label_to_list_freq_as_period_not_merge_consecutive():
    binary_series = pd.DataFrame(
        np.array(
            [
                [0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ).T,
        index=pd.date_range(start=0, periods=10, freq="1d"),
        columns=["type1", "type2", "type3", "type4"],
    )

    anomaly_list = to_events(
        binary_series, freq_as_period=True, merge_consecutive=False
    )

    anomaly_list_true = {
        "type1": [
            (
                Timestamp("1970-01-02 00:00:00"),
                Timestamp("1970-01-02 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-05 00:00:00"),
                Timestamp("1970-01-05 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-06 00:00:00"),
                Timestamp("1970-01-06 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-08 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-09 00:00:00"),
                Timestamp("1970-01-09 23:59:59.999999999"),
            ),
        ],
        "type2": [
            (
                Timestamp("1970-01-01 00:00:00"),
                Timestamp("1970-01-01 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-08 23:59:59.999999999"),

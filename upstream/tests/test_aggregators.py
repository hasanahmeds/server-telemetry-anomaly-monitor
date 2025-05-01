import pandas as pd
from pandas import Timestamp

import adtk.aggregator as aggt


def test_or_dict_of_lists():
    """
    Test OrAggregator with input as a dict of lists of time stamps or time
    stamp 2-tuples
    """
    lists = {
        "A": [
            (Timestamp("2017-1-1"), Timestamp("2017-1-2")),
            (Timestamp("2017-1-5"), Timestamp("2017-1-8")),
            Timestamp("2017-1-10"),
        ],
        "B": [
            Timestamp("2017-1-2"),
            (Timestamp("2017-1-3"), Timestamp("2017-1-6")),
            Timestamp("2017-1-8"),
            (Timestamp("2017-1-7"), Timestamp("2017-1-9")),
            (Timestamp("2017-1-11"), Timestamp("2017-1-11")),
        ],
    }
    assert aggt.OrAggregator().aggregate(lists) == [
        (Timestamp("2017-01-01 00:00:00"), Timestamp("2017-01-02 00:00:00")),
        (Timestamp("2017-01-03 00:00:00"), Timestamp("2017-01-09 00:00:00")),
        Timestamp("2017-1-10"),
        Timestamp("2017-1-11"),
    ]

    lists = {
        "A": [
            (Timestamp("2017-1-1"), Timestamp("2017-1-2")),
            (Timestamp("2017-1-5"), Timestamp("2017-1-8")),
            Timestamp("2017-1-10"),
        ],
        "B": [],
    }

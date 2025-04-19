"""Module is for data (time series and anomaly list) processing.
"""

from typing import Dict, List, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd


def validate_series(
    ts: Union[pd.Series, pd.DataFrame],
    check_freq: bool = True,
    check_categorical: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """Validate time series.

    This functoin will check some common critical issues of time series that
    may cause problems if anomaly detection is performed without fixing them.
    The function will automatically fix some of them and raise errors for the
    others.

    Issues will be checked and automatically fixed include:

    - Time index is not monotonically increasing;
    - Time index contains duplicated time stamps (fix by keeping first values);
    - (optional) Time index attribute `freq` is missed while the index follows
      a frequency;
    - (optional) Time series include categorical (non-binary) label columns
      (to fix by converting categorical labels into binary indicators).

    Issues will be checked and raise error include:

    - Wrong type of time series object (must be pandas Series or DataFrame);
    - Wrong type of time index object (must be pandas DatetimeIndex).

    Parameters
    ----------
    ts: pandas Series or DataFrame
        Time series to be validated.

    check_freq: bool, optional
        Whether to check time index attribute `freq` is missed. Default: True.

    check_categorical: bool, optional
        Whether to check time series include categorical (non-binary) label
        columns. Default: False.

    Returns
    -------
    pandas Series or DataFrame
        Validated time series.

    """

    ts = ts.copy()

    # check input type
    if not isinstance(ts, (pd.Series, pd.DataFrame)):
        raise TypeError("Input is not a pandas Series or DataFrame object")

    # check index type
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise TypeError(
            "Index of time series must be a pandas DatetimeIndex object."
        )

    # check duplicated
    if any(ts.index.duplicated(keep="first")):
        ts = ts[ts.index.duplicated(keep="first") == False]

    # check sorted
    if not ts.index.is_monotonic_increasing:
        ts.sort_index(inplace=True)

    # check time step frequency
    if check_freq:
        if (ts.index.freq is None) and (ts.index.inferred_freq is not None):
            ts = ts.asfreq(ts.index.inferred_freq)

    # convert categorical labels into binary indicators
    if check_categorical:
        if isinstance(ts, pd.DataFrame):
            ts = pd.get_dummies(ts)
        if isinstance(ts, pd.Series):
            seriesName = ts.name
            ts = pd.get_dummies(
                ts.to_frame(),
                prefix="" if seriesName is None else seriesName,
                prefix_sep="" if seriesName is None else "_",
            )
            if len(ts.columns) == 1:
                ts = ts[ts.columns[0]]
                ts.name = seriesName

    return ts


def validate_events(
    event_list: List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
    point_as_interval: bool = False,
) -> List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]:
    """Validate event list.

    This function will check and fix some common issues in an event list (a
    list of time windows), including invalid time window, overlapped time
    windows, unsorted events, etc.

    Parameters
    ----------
    event_list: list
        A list of events, where an event is a pandas Timestamp if it is
        instantaneous or a 2-tuple of pandas Timestamps if it is a closed time
        interval.

    point_as_interval: bool, optional
        Whether to return all instantaneous event as a close interval with
        identicial start point and end point. Default: False.

    Returns
    -------

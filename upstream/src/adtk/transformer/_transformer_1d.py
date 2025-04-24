"""Module for 1-dimensional transformers.

1-dimensional transformers transform 1-dimensional time series, i.e. pandas
Series, into different series, to extract useful information out of the
original time series.

"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels
from packaging.version import parse
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

from .._transformer_base import (
    _NonTrainableUnivariateTransformer,
    _TrainableUnivariateTransformer,
)
from .._utils import PandasBugError


class CustomizedTransformer1D(_TrainableUnivariateTransformer):
    """Univariate transformer derived from a user-given function and parameters.

    Parameters
    ----------
    transform_func: function
        A function transforming univariate time series.

        The first input argument must be a pandas Series, optional input
        argument may be accepted through parameter `transform_func_params` and
        the output of `fit_func`, and the output must be a pandas Series or
        DataFrame with the same index as input.

    transform_func_params: dict, optional
        Parameters of `transform_func`. Default: None.


"""Module for model pipeline and pipenet.

Pipeline or Pipenet connects multiple models (transformers, detectors, and/or
aggregators) into a "super" model that may perform complex anomaly detection
process.

"""
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from tabulate import tabulate

from .._aggregator_base import _Aggregator
from .._base import _Model, _TrainableModel
from .._detector_base import (  # _NonTrainableMultivariateDetector,
    _NonTrainableUnivariateDetector,
    _TrainableMultivariateDetector,
    _TrainableUnivariateDetector,
)
from .._transformer_base import (
    _NonTrainableMultivariateTransformer,
    _NonTrainableUnivariateTransformer,
    _TrainableMultivariateTransformer,
    _TrainableUnivariateTransformer,
)
from ..metrics import f1_score, iou, precision, recall

_Detector = (
    _NonTrainableUnivariateDetector,
    # _NonTrainableMultivariateDetector,
    _TrainableUnivariateDetector,
    _TrainableMultivariateDetector,
)
_Transformer = (
    _NonTrainableUnivariateTransformer,
    _NonTrainableMultivariateTransformer,
    _TrainableUnivariateTransformer,
    _TrainableMultivariateTransformer,
)


class Pipeline:
    """A Pipeline object chains transformers and a detector sequentially.

    Parameters
    ----------
    steps: list of 2-tuples (str, object)
        Components of this pipeline. Each 2-tuple represents a step in the
        pipeline (step name, model object).

    Examples
    --------
    >>> steps = [('moving average', RollingAggregate(agg='mean', window=10)),
                 ('filter quantile 0.99', QuantileAD(high=0.99))]
    >>> myPipeline = Pipeline(steps)

    """

    def __init__(self, steps: List[Tuple[str, _Model]]) -> None:
        self.steps = steps
        self._pipenet = Pipenet()
        self._update_internal_pipenet()

    def _update_internal_pipenet(self) -> None:
        pipenet_steps = dict()
        last_name = "original"
        for pipeline_step in self.steps:
            pipenet_steps.update(
                {
                    pipeline_step[0]: {
                        "model": pipeline_step[1],
                        "input": last_name,
                    }
                }
            )
            last_name = pipeline_step[0]
        self._pipenet.steps = pipenet_steps

    def fit(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        skip_fit: Optional[List[str]] = None,
        return_intermediate: bool = False,
    ) -> Optional[Dict[str, Optional[Union[pd.Series, pd.DataFrame]]]]:
        """Train all models in the pipeline sequentially.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series used to train models.

        skip_fit: list, optional
            Models to skip training. This could be used when pipeline contains
            models that are already trained by the same time series, and re-
            training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        dict, optional
            If return_intermediate=True, return intermediate results generated
            during training as a dictionary where keys are step names. If a
            step does not perform transformation or detection, the result of
            that step will be None.

        """
        self._update_internal_pipenet()
        return self._pipenet.fit(
            ts=ts, skip_fit=skip_fit, return_intermediate=return_intermediate
        )

    def detect(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        return_intermediate: bool = False,
        return_list: bool = False,
    ) -> Union[
        Union[
            pd.Series,
            pd.DataFrame,
            List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
        Dict[
            str,
            Union[
                pd.Series,
                pd.DataFrame,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
                Dict[
                    str,
                    List[
                        Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]
                    ],
                ],
            ],
        ],
    ]:
        """Transform time series sequentially along pipeline, and detect
        anomalies with the last detector.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        return_list: bool, optional
            Whether to return a list of anomalous events, or a binary series
            indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas Series, pandas DataFrame, list, or dict
            Detected anomalies.

            - If return_intermediate=False, return detected anomalies, i.e.
              result from last detector.
            - If return_intermediate=True, return results of all models in
              pipeline as a dict where each item represents the result of a
              model.
            - If return_list=False, result from a detector or an aggregator
              will be a binary pandas Series indicating normal/anomalous.
            - If return_list=True, result from a detector or an aggregator
              will be a list of events where an event is a pandas Timestamp if
              it is instantaneous or a 2-tuple of pandas Timestamps if it is a
              closed time interval.


        """
        self._update_internal_pipenet()
        return self._pipenet.detect(
            ts=ts,
            return_intermediate=return_intermediate,
            return_list=return_list,
        )

    def transform(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        return_intermediate: bool = False,
    ) -> Union[
        Union[pd.Series, pd.DataFrame],
        Dict[str, Union[pd.Series, pd.DataFrame]],
    ]:
        """Transform time series sequentially along pipeline.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to be transformed

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        pandas Series, pandas DataFrame, or dict
            Transformed time series.

            - If return_intermediate=False, return transformed series, i.e.
              result from last transformer;
            - If return_intermediate=True, return results of all models in
              pipeline as a dict where each item represents the result of a
              model.

        """
        self._update_internal_pipenet()
        return self._pipenet.transform(
            ts=ts, return_intermediate=return_intermediate
        )

    def fit_detect(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        skip_fit: Optional[List[str]] = None,
        return_intermediate: bool = False,
        return_list: bool = False,
    ) -> Union[
        Union[
            pd.Series,
            pd.DataFrame,
            List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
        Dict[
            str,
            Union[
                pd.Series,
                pd.DataFrame,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
                Dict[
                    str,
                    List[
                        Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]
                    ],
                ],
            ],
        ],
    ]:
        """Train models in pipeline sequentially, transform time series along
        pipeline, and use the last detector to detect anomalies.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.

        skip_fit: list, optional
            Models to skip training. This could be used when pipeline contains
            models that are already trained by the same time series, and re-
            training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        return_list: bool, optional
            Whether to return a list of anomalous events, or a binary series
            indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas Series, pandas DataFrame, list, or dict
            Detected anomalies.

            - If return_intermediate=False, return detected anomalies, i.e.
              result from last detector.
            - If return_intermediate=True, return results of all models in
              pipeline as a dict where each item represents the result of a
              model.
            - If return_list=False, result from a detector or an aggregator
              will be a binary pandas Series indicating normal/anomalous.
            - If return_list=True, result from a detector or an aggregator
              will be a list of events where an event is a pandas Timestamp if
              it is instantaneous or a 2-tuple of pandas Timestamps if it is a
              closed time interval.


        """
        self._update_internal_pipenet()
        return self._pipenet.fit_detect(
            ts=ts,
            skip_fit=skip_fit,
            return_intermediate=return_intermediate,
            return_list=return_list,
        )

    def fit_transform(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        skip_fit: Optional[List[str]] = None,
        return_intermediate: bool = False,
    ) -> Union[
        Union[pd.Series, pd.DataFrame],
        Dict[str, Union[pd.Series, pd.DataFrame]],
    ]:
        """Train models in pipeline sequentially, and transform time series
        along pipeline.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to be transformed.

        skip_fit: list, optional
            Models to skip training. This could be used when pipeline contains
            models that are already trained by the same time series, and re-
            training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        pandas Series, pandas DataFrame, or dict
            Transformed time series.

            - If return_intermediate=False, return transformed series, i.e.
              result from last transformer;
            - If return_intermediate=True, return results of all models in
              pipeline as a dict where each item represents the result of a
              model.

        """
        self._update_internal_pipenet()
        return self._pipenet.fit_transform(
            ts=ts, skip_fit=skip_fit, return_intermediate=return_intermediate
        )

    def score(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        anomaly_true: Union[
            pd.Series,
            pd.DataFrame,
            List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
        scoring: str = "recall",
        **kwargs: Any
    ) -> Union[float, Dict[str, float]]:
        """Detect anomalies and score the results against true anomalies.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.

        anomaly_true: pandas Series or list
            True anomalies.

            - If pandas Series, it is treated as a series of binary labels.
            - If list, a list of events where an event is a pandas Timestamp if
              it is instantaneous or a 2-tuple of pandas Timestamps if it is a
              closed time interval.

        scoring: str, optional
            Scoring function to use. Must be one of "recall", "precision",
            "f1", and "iou". See module `metrics` for more information.
            Default: "recall"

        **kwargs
            Optional parameters for scoring function. See module `metrics` for
            more information.

        Returns
        -------
        float
            Score of detection result.

        """
        if scoring == "recall":
            scoring_func = recall  # type: Callable
        elif scoring == "precision":
            scoring_func = precision
        elif scoring == "f1":
            scoring_func = f1_score
        elif scoring == "iou":
            scoring_func = iou
        else:
            raise ValueError(
                "Argument `scoring` must be one of 'recall', 'precision', "
                "'f1' and 'iou'."
            )
        if isinstance(anomaly_true, pd.Series):
            return scoring_func(
                y_true=anomaly_true,
                y_pred=self.detect(ts, return_list=False),
                **kwargs
            )
        else:
            return scoring_func(
                y_true=anomaly_true,
                y_pred=self.detect(ts, return_list=True),
                **kwargs
            )

    def get_params(self) -> Dict[str, Dict[str, Any]]:
        """Get parameters of models in pipeline.

        Returns
        -------
        dict
            A dictionary of model name and model parameters.

        """
        self._update_internal_pipenet()
        return self._pipenet.get_params()


class Pipenet:
    """A Pipenet object connects transformers, detectors and aggregators.

    Parameters
    ----------
    steps: dicts
        Components of the pipenet. Each key-value item represents a step (
        transformer, detector, or aggregator), where key is the unique name of
        the step and the value is a dict with the following key-value pairs:

            - input (str or list of str): Input to the model, which must be
              either 'original' (i.e. the input time series), or the name of
              a upstream component.
            - subset (str, list of str, or list of lists of str, optional): If
              a model does not use all series from an input component, use this
              field to specify which series should be included. If not given or
              "all", all series from the input component will be used.
            - model (object): A detector, transformer, or aggregator object.

    Attributes
    ----------
    steps_graph_: OrderedDict
        Order of steps to be executed. Keys are step names, values are 2-tuple
        (i, j) where i is the index of execution round and j is the the index
        within a round.

    final_step_: str
        Name of the final step to be executed. It is the single step in the
        last round of execution in attribute `steps_graph_`.

    Examples
    --------
    The following example show how to use a Pipenet to build a level shift
    detector with some basic transformers, detectors, and aggregator.

    >>> from adtk.detector import QuantileAD, ThresholdAD
    >>> from adtk.transformer import DoubleRollingAggregate
    >>> from adtk.aggregator import AndAggregator
    >>> from adtk.pipe import Pipenet
    >>> steps = {
            "diff_abs": {
                "input": "original",
                "model": DoubleRollingAggregate(
                    agg="median",
                    window=20,
                    center=True,
                    diff="l1",
                ),
            },
            "quantile_ad": {
                "input": "diff_abs",
                "model": QuantileAD(high=0.99, low=0),
            },
            "diff": {
                "input": "original",
                "model": DoubleRollingAggregate(
                    agg="median",
                    window=20,
                    center=True,
                    diff="diff",
                ),
            },
            "sign_check": {
                "input": "diff",
                "model": ThresholdAD(high=0.0, low=-float("inf")),
            },
            "and": {
                "model": AndAggregator(),
                "input": ["quantile_ad", "sign_check"],
            },
        }
    >>> myPipenet = Pipenet(steps)

    """

    def __init__(
        self, steps: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        if steps is None:
            self.steps = dict()
        else:
            self.steps = steps
        self._validate()

    def _validate(self) -> None:
        """
        Check the following issues and raise error if found
        - steps is not a list of dict
        - some step does not have required keys

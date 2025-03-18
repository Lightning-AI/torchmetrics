# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from functools import partial
from typing import Optional

import numpy as np
import pytest
import torch
from permetrics.regression import RegressionMetric
from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error as sk_mean_abs_percentage_error
from sklearn.metrics import mean_squared_error as sk_mean_squared_error
from sklearn.metrics import mean_squared_log_error as sk_mean_squared_log_error
from sklearn.metrics._regression import _check_reg_targets
from sklearn.utils import check_consistent_length

from torchmetrics.functional import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    normalized_root_mean_squared_error,
    weighted_mean_absolute_percentage_error,
)
from torchmetrics.functional.regression.symmetric_mape import symmetric_mean_absolute_percentage_error
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogError,
    WeightedMeanAbsolutePercentageError,
)
from torchmetrics.regression.nrmse import NormalizedRootMeanSquaredError
from torchmetrics.regression.symmetric_mape import SymmetricMeanAbsolutePercentageError
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)

NUM_TARGETS = 5


_single_target_inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_multi_target_inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
)


def _reference_symmetric_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    multioutput: str = "uniform_average",
):
    r"""Symmetric mean absolute percentage error regression loss (SMAPE_).

    .. math:: \text{SMAPE} = \frac{2}{n}\sum_1^n\frac{max(|   y_i - \hat{y_i} |}{| y_i | + | \hat{y_i} |, \epsilon)}

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Args:
        y_true: array-like of shape (n_samples,) or (n_samples, n_outputs)
            Ground truth (correct) target values.
        y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
            Estimated target values.
        sample_weight: array-like of shape (n_samples,), default=None
            Sample weights.
        multioutput: {'raw_values', 'uniform_average'} or array-like
            Defines aggregating of multiple output values.
            Array-like value defines weights used to average errors.
            If input is list then the shape must be (n_outputs,).

                - 'raw_values': Returns a full set of errors in case of multioutput input.
                - 'uniform_average': Errors of all outputs are averaged with uniform weight.

    Returns:
        loss: float or ndarray of floats in the range [0, 1]
            If multi-output is 'raw_values', then symmetric mean absolute percentage error
            is returned for each output separately.
            If multi-output is 'uniform_average' or a ndarray of weights, then the
            weighted average of all output errors is returned.
            MAPE output is non-negative floating point. The best value is 0.0.
            But note the fact that bad predictions can lead to arbitrarily large
            MAPE values, especially if some y_true values are very close to zero.
            Note that we return a large value instead of `inf` when y_true is zero.

    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    epsilon = np.finfo(np.float64).eps
    smape = 2 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), epsilon)
    output_errors = np.average(smape, weights=sample_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        # pass None as weights to np.average: uniform mean
        multioutput = None

    return np.average(output_errors, weights=multioutput)


def _reference_normalized_root_mean_squared_error(
    y_true: np.ndarray, y_pred: np.ndarray, normalization: str = "mean", num_outputs: int = 1
):
    """Reference implementation of Normalized Root Mean Squared Error (NRMSE) metric."""
    if num_outputs == 1:
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
    if normalization != "l2":
        evaluator = RegressionMetric(y_true, y_pred) if normalization == "range" else RegressionMetric(y_pred, y_true)
        arg_mapping = {"mean": 1, "range": 2, "std": 4}
        return evaluator.normalized_root_mean_square_error(model=arg_mapping[normalization])
    # for l2 normalization we do not have a reference implementation
    return np.sqrt(np.mean(np.square(y_true - y_pred), axis=0)) / np.linalg.norm(y_true, axis=0)


def _reference_weighted_mean_abs_percentage_error(target, preds):
    """Reference implementation of Weighted Mean Absolute Percentage Error (WMAPE) metric."""
    return np.sum(np.abs(target - preds)) / np.sum(np.abs(target))


def _single_target_ref_wrapper(preds, target, sk_fn, metric_args):
    """Reference implementation of single-target metrics."""
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    if metric_args and "normalization" in metric_args:
        res = sk_fn(sk_target, sk_preds, normalization=metric_args["normalization"])
    else:
        res = sk_fn(sk_target, sk_preds)
    if metric_args and "squared" in metric_args and not metric_args["squared"]:
        res = math.sqrt(res)
    return res


def _multi_target_ref_wrapper(preds, target, sk_fn, metric_args):
    """Reference implementation of multi-target metrics."""
    sk_preds = preds.view(-1, NUM_TARGETS).numpy()
    sk_target = target.view(-1, NUM_TARGETS).numpy()
    sk_kwargs = {"multioutput": "raw_values"} if metric_args and "num_outputs" in metric_args else {}
    if metric_args and "normalization" in metric_args:
        res = sk_fn(sk_target, sk_preds, **metric_args)
    else:
        res = sk_fn(sk_target, sk_preds, **sk_kwargs)
    if metric_args and "squared" in metric_args and not metric_args["squared"]:
        res = math.sqrt(res)
    return res


@pytest.mark.parametrize(
    ("preds", "target", "ref_metric"),
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_ref_wrapper),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_ref_wrapper),
    ],
)
@pytest.mark.parametrize(
    ("metric_class", "metric_functional", "sk_fn", "metric_args"),
    [
        pytest.param(
            MeanSquaredError, mean_squared_error, sk_mean_squared_error, {"squared": True}, id="mse_singleoutput"
        ),
        pytest.param(
            MeanSquaredError, mean_squared_error, sk_mean_squared_error, {"squared": False}, id="rmse_singleoutput"
        ),
        pytest.param(
            MeanSquaredError,
            mean_squared_error,
            sk_mean_squared_error,
            {"squared": True, "num_outputs": NUM_TARGETS},
            id="mse_multioutput",
        ),
        pytest.param(MeanAbsoluteError, mean_absolute_error, sk_mean_absolute_error, {}, id="mae_singleoutput"),
        pytest.param(
            MeanAbsoluteError,
            mean_absolute_error,
            sk_mean_absolute_error,
            {"num_outputs": NUM_TARGETS},
            id="mae_multioutput",
        ),
        pytest.param(
            MeanAbsolutePercentageError,
            mean_absolute_percentage_error,
            sk_mean_abs_percentage_error,
            {},
            id="mape_singleoutput",
        ),
        pytest.param(
            SymmetricMeanAbsolutePercentageError,
            symmetric_mean_absolute_percentage_error,
            _reference_symmetric_mape,
            {},
            id="symmetric_mean_absolute_percentage_error",
        ),
        pytest.param(
            MeanSquaredLogError, mean_squared_log_error, sk_mean_squared_log_error, {}, id="mean_squared_log_error"
        ),
        pytest.param(
            WeightedMeanAbsolutePercentageError,
            weighted_mean_absolute_percentage_error,
            _reference_weighted_mean_abs_percentage_error,
            {},
            id="weighted_mean_absolute_percentage_error",
        ),
        pytest.param(
            NormalizedRootMeanSquaredError,
            normalized_root_mean_squared_error,
            _reference_normalized_root_mean_squared_error,
            {"normalization": "mean", "num_outputs": 1},
            id="nrmse_singleoutput_mean",
        ),
        pytest.param(
            NormalizedRootMeanSquaredError,
            normalized_root_mean_squared_error,
            _reference_normalized_root_mean_squared_error,
            {"normalization": "range", "num_outputs": 1},
            id="nrmse_singleoutput_range",
        ),
        pytest.param(
            NormalizedRootMeanSquaredError,
            normalized_root_mean_squared_error,
            _reference_normalized_root_mean_squared_error,
            {"normalization": "std", "num_outputs": 1},
            id="nrmse_singleoutput_std",
        ),
        pytest.param(
            NormalizedRootMeanSquaredError,
            normalized_root_mean_squared_error,
            _reference_normalized_root_mean_squared_error,
            {"normalization": "l2", "num_outputs": 1},
            id="nrmse_multioutput_l2",
        ),
        pytest.param(
            NormalizedRootMeanSquaredError,
            normalized_root_mean_squared_error,
            _reference_normalized_root_mean_squared_error,
            {"normalization": "mean", "num_outputs": NUM_TARGETS},
            id="nrmse_multioutput_mean",
        ),
        pytest.param(
            NormalizedRootMeanSquaredError,
            normalized_root_mean_squared_error,
            _reference_normalized_root_mean_squared_error,
            {"normalization": "range", "num_outputs": NUM_TARGETS},
            id="nrmse_multioutput_range",
        ),
        pytest.param(
            NormalizedRootMeanSquaredError,
            normalized_root_mean_squared_error,
            _reference_normalized_root_mean_squared_error,
            {"normalization": "std", "num_outputs": NUM_TARGETS},
            id="nrmse_multioutput_std",
        ),
        pytest.param(
            NormalizedRootMeanSquaredError,
            normalized_root_mean_squared_error,
            _reference_normalized_root_mean_squared_error,
            {"normalization": "l2", "num_outputs": NUM_TARGETS},
            id="nrmse_multioutput_l2",
        ),
    ],
)
class TestMeanError(MetricTester):
    """Test class for `MeanError` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_mean_error_class(
        self, preds, target, ref_metric, metric_class, metric_functional, sk_fn, metric_args, ddp
    ):
        """Test class implementation of metric."""
        if metric_args and "num_outputs" in metric_args and preds.ndim < 3:
            pytest.skip("Test only runs for multi-output setting")
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            reference_metric=partial(ref_metric, sk_fn=sk_fn, metric_args=metric_args),
            metric_args=metric_args,
        )

    def test_mean_error_functional(
        self, preds, target, ref_metric, metric_class, metric_functional, sk_fn, metric_args
    ):
        """Test functional implementation of metric."""
        if metric_args and "num_outputs" in metric_args and preds.ndim < 3:
            pytest.skip("Test only runs for multi-output setting")
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=metric_functional,
            reference_metric=partial(ref_metric, sk_fn=sk_fn, metric_args=metric_args),
            metric_args=metric_args,
        )

    def test_mean_error_differentiability(
        self, preds, target, ref_metric, metric_class, metric_functional, sk_fn, metric_args
    ):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        if metric_args and "num_outputs" in metric_args and preds.ndim < 3:
            pytest.skip("Test only runs for multi-output setting")
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=metric_class,
            metric_functional=metric_functional,
            metric_args=metric_args,
        )

    def test_mean_error_half_cpu(self, preds, target, ref_metric, metric_class, metric_functional, sk_fn, metric_args):
        """Test dtype support of the metric on CPU."""
        if metric_class == MeanSquaredLogError:
            # MeanSquaredLogError half + cpu does not work due to missing support in torch.log
            pytest.xfail("MeanSquaredLogError metric does not support cpu + half precision")

        if metric_class == MeanAbsolutePercentageError:
            # MeanSquaredPercentageError half + cpu does not work due to missing support in torch.log
            pytest.xfail("MeanSquaredPercentageError metric does not support cpu + half precision")

        if metric_class == SymmetricMeanAbsolutePercentageError:
            # MeanSquaredPercentageError half + cpu does not work due to missing support in torch.log
            pytest.xfail("SymmetricMeanAbsolutePercentageError metric does not support cpu + half precision")

        if metric_class == WeightedMeanAbsolutePercentageError:
            # WeightedMeanAbsolutePercentageError half + cpu does not work due to missing support in torch.clamp
            pytest.xfail("WeightedMeanAbsolutePercentageError metric does not support cpu + half precision")

        if metric_class == NormalizedRootMeanSquaredError:
            # NormalizedRootMeanSquaredError half + cpu does not work due to missing support in torch.sqrt
            pytest.xfail("NormalizedRootMeanSquaredError metric does not support cpu + half precision")

        self.run_precision_test_cpu(preds, target, metric_class, metric_functional)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_mean_error_half_gpu(self, preds, target, ref_metric, metric_class, metric_functional, sk_fn, metric_args):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(preds, target, metric_class, metric_functional)


@pytest.mark.parametrize(
    "metric_class",
    [
        MeanSquaredError,
        MeanAbsoluteError,
        MeanSquaredLogError,
        MeanAbsolutePercentageError,
        NormalizedRootMeanSquaredError,
    ],
)
def test_error_on_different_shape(metric_class):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))


@pytest.mark.parametrize(
    ("metric_class", "arguments", "error_msg"),
    [
        (MeanSquaredError, {"squared": "something"}, "Expected argument `squared` to be a boolean.*"),
        (NormalizedRootMeanSquaredError, {"normalization": "something"}, "Argument `normalization` should be either.*"),
    ],
)
def test_error_on_wrong_extra_args(metric_class, arguments, error_msg):
    """Test that error is raised on wrong extra arguments."""
    with pytest.raises(ValueError, match=error_msg):
        metric_class(**arguments)

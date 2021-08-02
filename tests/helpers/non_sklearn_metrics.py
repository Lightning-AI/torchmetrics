"""File for non sklearn metrics that are to be used for reference for tests"""
from typing import Optional

import numpy as np
from sklearn.metrics._regression import _check_reg_targets
from sklearn.utils.validation import check_consistent_length


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    multioutput: str = "uniform_average",
):
    r"""Symmetric mean absolute percentage error regression loss.
    <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>`_ (SMAPE):

    .. math:: \text{SMAPE} = \frac{2}{n}\sum_1^n\frac{max(|   y_i - \hat{y_i} |}{| y_i | + | \hat{y_i} |, \epsilon)}

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    multioutput : {'raw_values', 'uniform_average'} or array-like
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If input is list then the shape must be (n_outputs,).
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    Returns
    -------
    loss : float or ndarray of floats in the range [0, 1]
        If multioutput is 'raw_values', then symmetric mean absolute percentage error
        is returned for each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average of all output errors is returned.
        MAPE output is non-negative floating point. The best value is 0.0.
        But note the fact that bad predictions can lead to arbitarily large
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

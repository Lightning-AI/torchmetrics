import numpy as np
from sklearn.utils import assert_all_finite, check_consistent_length, column_or_1d

# sklearn reference function from
# https://github.com/samronsin/scikit-learn/blob/calibration-loss/sklearn/metrics/_classification.py.
# TODO: when the PR into sklearn is accepted, update this to use the official function.


def calibration_error(y_true, y_prob, sample_weight=None, norm='l2',
                      n_bins=10, strategy='uniform', pos_label=None,
                      reduce_bias=True):
    """Compute calibration error of a binary classifier.
    Across all items in a set of N predictions, the calibration error measures
    the aggregated difference between (1) the average predicted probabilities
    assigned to the positive class, and (2) the frequencies
    of the positive class in the actual outcome.
    The calibration error is only appropriate for binary categorical outcomes.
    Which label is considered to be the positive label is controlled via the
    parameter pos_label, which defaults to 1.
    Read more in the :ref:`User Guide <calibration>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets of a binary classification task.
    y_prob : array-like of (n_samples,)
        Probabilities of the positive class.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    norm : {'l1', 'l2', 'max'}, default='l2'
        Norm method. The l1-norm is the Expected Calibration Error (ECE),
        and the max-norm corresponds to Maximum Calibration Error (MCE).
    n_bins : int, default=10
       The number of bins to compute error on.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.
        uniform
            All bins have identical widths.
        quantile
            All bins have the same number of points.
    pos_label : int or str, default=None
        Label of the positive class. If None, the maximum label is used as
        positive class.
    reduce_bias : bool, default=True
        Add debiasing term as in Verified Uncertainty Calibration, A. Kumar.
        Only effective for the l2-norm.
    Returns
    -------
    score : float
        calibration error
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import calibration_error
    >>> y_true = np.array([0, 0, 0, 1] + [0, 1, 1, 1])
    >>> y_pred = np.array([0.25, 0.25, 0.25, 0.25] + [0.75, 0.75, 0.75, 0.75])
    >>> calibration_error(y_true, y_pred, n_bins=5)
    0.0
    >>> calibration_error(
    ...     y_true, y_pred, n_bins=5, norm="max"
    ... )
    0.0
    >>> y_true = np.array([0, 0, 0, 0] + [1, 1, 1, 1])
    >>> calibration_error(y_true, y_pred, n_bins=5)
    0.25
    >>> calibration_error(
    ...     y_true, y_pred, n_bins=5, norm="max"
    ... )
    0.25
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    assert_all_finite(y_true)
    assert_all_finite(y_prob)
    check_consistent_length(y_true, y_prob, sample_weight)
    if any(y_prob < 0) or any(y_prob > 1):
        raise ValueError("y_prob has values outside of [0, 1] range")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError("Only binary classification is supported. "
                         "Provided labels %s." % labels)

    if pos_label is None:
        pos_label = y_true.max()
    if pos_label not in labels:
        raise ValueError("pos_label=%r is not a valid label: "
                         "%r" % (pos_label, labels))
    y_true = np.array(y_true == pos_label, int)

    norm_options = ('l1', 'l2', 'max')
    if norm not in norm_options:
        raise ValueError(f'norm has to be one of {norm_options}, got: {norm}.')

    remapping = np.argsort(y_prob)
    y_true = y_true[remapping]
    y_prob = y_prob[remapping]
    if sample_weight is not None:
        sample_weight = sample_weight[remapping]
    else:
        sample_weight = np.ones(y_true.shape[0])

    n_bins = int(n_bins)
    if strategy == 'quantile':
        quantiles = np.percentile(y_prob, np.arange(0, 1, 1.0 / n_bins) * 100)
    elif strategy == 'uniform':
        quantiles = np.arange(0, 1, 1.0 / n_bins)
    else:
        raise ValueError(
            f"Invalid entry to 'strategy' input. Strategy must be either "
            f"'quantile' or 'uniform'. Got {strategy} instead."
        )

    threshold_indices = np.searchsorted(y_prob, quantiles).tolist()
    threshold_indices.append(y_true.shape[0])
    avg_pred_true = np.zeros(n_bins)
    bin_centroid = np.zeros(n_bins)
    delta_count = np.zeros(n_bins)
    debias = np.zeros(n_bins)

    loss = 0.
    count = float(sample_weight.sum())
    for i, i_start in enumerate(threshold_indices[:-1]):
        i_end = threshold_indices[i + 1]
        # ignore empty bins
        if i_end == i_start:
            continue
        delta_count[i] = float(sample_weight[i_start:i_end].sum())
        avg_pred_true[i] = (np.dot(y_true[i_start:i_end],
                                   sample_weight[i_start:i_end]) /
                            delta_count[i])
        bin_centroid[i] = (np.dot(y_prob[i_start:i_end],
                                  sample_weight[i_start:i_end]) /
                           delta_count[i])
        if norm == "l2" and reduce_bias:
            # NOTE: I think there's a mistake in the original implementation.
            # delta_debias = (
            #     avg_pred_true[i] * (avg_pred_true[i] - 1) * delta_count[i]
            # )
            # delta_debias /= (count * delta_count[i] - 1)
            delta_debias = (
                avg_pred_true[i] * (avg_pred_true[i] - 1) * delta_count[i]
            )
            delta_debias /= count * (delta_count[i] - 1)
            debias[i] = delta_debias

    if norm == "max":
        loss = np.max(np.abs(avg_pred_true - bin_centroid))
    elif norm == "l1":
        delta_loss = np.abs(avg_pred_true - bin_centroid) * delta_count
        loss = np.sum(delta_loss) / count
    elif norm == "l2":
        delta_loss = (avg_pred_true - bin_centroid)**2 * delta_count
        loss = np.sum(delta_loss) / count
        if reduce_bias:
            # convert nans to zero
            loss += np.sum(np.nan_to_num(debias))
        loss = np.sqrt(max(loss, 0.))
    return loss

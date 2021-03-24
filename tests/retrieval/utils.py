from typing import Callable, List

import numpy as np
from torch import Tensor


def _compute_sklearn_metric(
    metric: Callable, target: List[np.ndarray], preds: List[np.ndarray], behaviour: str
) -> Tensor:
    """ Compute metric with multiple iterations over documents predictions. """
    sk_results = []

    for b, a in zip(target, preds):
        if b.sum() == 0:
            if behaviour == 'skip':
                pass
            elif behaviour == 'pos':
                sk_results.append(1.0)
            else:
                sk_results.append(0.0)
        else:
            res = metric(b, a)
            sk_results.append(res)

    if len(sk_results) > 0:
        return np.mean(sk_results)
    return np.array(0.0)


def _reciprocal_rank(target: np.array, preds: np.array):
    """
    Implementation of reciprocal rank because couldn't find a good implementation.
    `sklearn.metrics.label_ranking_average_precision_score` is similar but works in a different way
    then the number of positive labels is greater than 1.
    """
    assert target.shape == preds.shape
    assert len(target.shape) == 1  # works only with single dimension inputs

    if target.sum() > 0:
        target = target[np.argsort(preds, axis=-1)][::-1]
        rank = np.nonzero(target)[0][0] + 1
        return 1.0 / rank
    else:
        return np.NaN


def _assert_error(function, error, *args, **kwargs):
    """ Assert that `function(*args, **kwargs)` raises `error`. """
    try:
        function(*args, **kwargs)
        assert False  # assert exception is raised
    except Exception as e:
        assert isinstance(e, error)

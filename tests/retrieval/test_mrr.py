import numpy as np
import pytest
from sklearn.metrics import label_ranking_average_precision_score

from tests.retrieval.helpers import _test_dtypes, _test_input_shapes, _test_retrieval_against_sklearn
from torchmetrics.retrieval.mean_reciprocal_rank import RetrievalMRR


def _reciprocal_rank(target: np.array, preds: np.array):
    """
    Adaptation of `sklearn.metrics.label_ranking_average_precision_score`.
    Since the original sklearn metric works as RR only when the number of positive
    targets is exactly 1, here we remove every positive target that is not the most
    important. Remember that in RR only the positive target with the highest score is considered.
    """
    assert target.shape == preds.shape
    assert len(target.shape) == 1  # works only with single dimension inputs

    # going to remove T targets that are not ranked as highest
    indexes = preds[target.astype(np.bool)]
    if len(indexes) > 0:
        target[preds != indexes.max(-1, keepdims=True)[0]] = 0  # ensure that only 1 positive label is present

    if target.sum() > 0:
        # sklearn `label_ranking_average_precision_score` requires at most 2 dims
        return label_ranking_average_precision_score(np.expand_dims(target, axis=0), np.expand_dims(preds, axis=0))
    else:
        return 0.0


@pytest.mark.parametrize('size', [1, 4, 10])
@pytest.mark.parametrize('n_documents', [1, 5])
@pytest.mark.parametrize('empty_target_action', ['skip', 'pos', 'neg'])
def test_results(size, n_documents, empty_target_action):
    """ Test metrics are computed correctly. """
    _test_retrieval_against_sklearn(_reciprocal_rank, RetrievalMRR, size, n_documents, empty_target_action)


def test_dtypes():
    """ Check dypes are managed correctly. """
    _test_dtypes(RetrievalMRR)


def test_input_shapes() -> None:
    """Check inputs shapes are managed correctly. """
    _test_input_shapes(RetrievalMRR)

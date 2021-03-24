import numpy as np
import pytest

from tests.retrieval.helpers import _test_against_sklearn, _test_dtypes, _test_input_shapes
from torchmetrics.retrieval.mean_reciprocal_rank import RetrievalMRR


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


@pytest.mark.parametrize('size', [1, 4, 10])
@pytest.mark.parametrize('n_documents', [1, 5])
@pytest.mark.parametrize('query_without_relevant_docs_options', ['skip', 'pos', 'neg'])
def test_results(size, n_documents, query_without_relevant_docs_options):
    """ Test metrics are computed correctly. """
    _test_against_sklearn(
        _reciprocal_rank,
        RetrievalMRR,
        size,
        n_documents,
        query_without_relevant_docs_options
    )


def test_dtypes():
    """ Check dypes are managed correctly. """
    _test_dtypes(RetrievalMRR)


def test_input_shapes() -> None:
    """Check inputs shapes are managed correctly. """
    _test_input_shapes(RetrievalMRR)

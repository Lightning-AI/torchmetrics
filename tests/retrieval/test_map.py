import pytest
from sklearn.metrics import average_precision_score as sk_average_precision

from tests.retrieval.helpers import _test_dtypes, _test_input_shapes, _test_retrieval_against_sklearn
from torchmetrics.retrieval.mean_average_precision import RetrievalMAP


@pytest.mark.parametrize('size', [1, 4, 10])
@pytest.mark.parametrize('n_documents', [1, 5])
@pytest.mark.parametrize('empty_target_action', ['skip', 'pos', 'neg'])
def test_results(size, n_documents, empty_target_action):
    """ Test metrics are computed correctly. """
    _test_retrieval_against_sklearn(
        sk_average_precision, RetrievalMAP, size, n_documents, empty_target_action
    )


def test_dtypes():
    """ Check dypes are managed correctly. """
    _test_dtypes(RetrievalMAP)


def test_input_shapes() -> None:
    """Check inputs shapes are managed correctly. """
    _test_input_shapes(RetrievalMAP)

import numpy as np
import pytest
import torch
from pytorch_lightning import seed_everything
from sklearn.metrics import average_precision_score as sk_average_precision

from tests.retrieval.helpers import _assert_error, _compute_sklearn_metric
from tests.retrieval.helpers import _reciprocal_rank as reciprocal_rank
from torchmetrics.retrieval.mean_average_precision import RetrievalMAP
from torchmetrics.retrieval.mean_reciprocal_rank import RetrievalMRR


@pytest.mark.parametrize(['sklearn_metric', 'torch_metric'], [
    [sk_average_precision, RetrievalMAP],
    [reciprocal_rank, RetrievalMRR],
])
@pytest.mark.parametrize('size', [1, 4, 10])
@pytest.mark.parametrize('rounds', [20])
@pytest.mark.parametrize('n_documents', [1, 5])
@pytest.mark.parametrize('query_without_relevant_docs_options', ['skip', 'pos', 'neg'])
def test_against_sklearn(
    sklearn_metric, torch_metric, size, rounds, n_documents, query_without_relevant_docs_options
) -> None:
    """ Compare PL metrics to standard version. """
    seed_everything(0)

    for _ in range(rounds):

        metric = torch_metric(query_without_relevant_docs=query_without_relevant_docs_options)
        shape = (size, )

        indexes = []
        preds = []
        target = []

        for i in range(n_documents):
            indexes.append(np.ones(shape, dtype=int) * i)
            preds.append(np.random.randn(*shape))
            target.append(np.random.randn(*shape) > 0)

        sk_results = _compute_sklearn_metric(sklearn_metric, target, preds, query_without_relevant_docs_options)
        sk_results = torch.tensor(sk_results)

        indexes_tensor = torch.cat([torch.tensor(i) for i in indexes])
        preds_tensor = torch.cat([torch.tensor(p) for p in preds])
        target_tensor = torch.cat([torch.tensor(t) for t in target])

        # lets assume data are not ordered
        perm = torch.randperm(indexes_tensor.nelement())
        indexes_tensor = indexes_tensor.view(-1)[perm].view(indexes_tensor.size())
        preds_tensor = preds_tensor.view(-1)[perm].view(preds_tensor.size())
        target_tensor = target_tensor.view(-1)[perm].view(target_tensor.size())

        # shuffle ids to require also sorting of documents ability from the lightning metric
        pl_result = metric(indexes_tensor, preds_tensor, target_tensor)

        assert torch.allclose(sk_results.float(), pl_result.float(), equal_nan=False), (
            f"Test failed comparing metric {sklearn_metric} with {torch_metric}: "
            f"{sk_results.float()} vs {pl_result.float()}. "
            f"indexes: {indexes}, preds: {preds}, target: {target}"
        )


@pytest.mark.parametrize('torchmetric', [RetrievalMAP, RetrievalMRR])
def test_input_dtypes(torchmetric) -> None:
    """Check PL metrics inputs are controlled correctly. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(0)

    length = 10  # not important in this test

    # check error when `query_without_relevant_docs='error'` is raised correctly
    indexes = torch.tensor([0] * length, device=device, dtype=torch.int64)
    preds = torch.rand(size=(length, ), device=device, dtype=torch.float32)
    target = torch.tensor([False] * length, device=device, dtype=torch.bool)

    metric = torchmetric(query_without_relevant_docs='error')
    _assert_error(metric, ValueError, indexes, preds, target)

    # check ValueError with invalid `query_without_relevant_docs` argument
    _assert_error(torchmetric, ValueError, query_without_relevant_docs='casual_argument')

    # check input dtypes
    indexes = torch.tensor([0] * length, device=device, dtype=torch.int64)
    preds = torch.tensor([0] * length, device=device, dtype=torch.float32)
    target = torch.tensor([0] * length, device=device, dtype=torch.int64)

    metric = torchmetric(query_without_relevant_docs='error')

    # check error on input dtypes are raised correctly
    _assert_error(metric, ValueError, indexes.bool(), preds, target)
    _assert_error(metric, ValueError, indexes, preds.bool(), target)
    _assert_error(metric, ValueError, indexes, preds, target.float())


@pytest.mark.parametrize('torchmetric', [RetrievalMAP, RetrievalMRR])
def test_input_shapes(torchmetric) -> None:
    """Check PL metrics inputs are controlled correctly. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(0)

    metric = torchmetric(query_without_relevant_docs='error')

    # check input shapes are checked correclty
    for _ in range(10):
        # ensure sizes are different
        elements_1, elements_2, elements_3 = np.random.choice(20, size=3, replace=False)
        indexes = torch.tensor([0] * elements_1, device=device, dtype=torch.int64)
        preds = torch.tensor([0] * elements_2, device=device, dtype=torch.float32)
        target = torch.tensor([0] * elements_3, device=device, dtype=torch.int64)

        _assert_error(metric, ValueError, indexes, preds, target)

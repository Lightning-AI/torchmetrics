from typing import Callable, List

import numpy as np
import pytest
import torch
from torch import Tensor

from tests.helpers import seed_all
from torchmetrics import Metric

seed_all(1337)


def _compute_sklearn_metric(
    metric: Callable, target: List[np.ndarray], preds: List[np.ndarray], behaviour: str, **kwargs
) -> Tensor:
    """ Compute metric with multiple iterations over every query predictions set. """
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
            res = metric(b, a, **kwargs)
            sk_results.append(res)

    if len(sk_results) > 0:
        return np.mean(sk_results)
    return np.array(0.0)


def _test_retrieval_against_sklearn(
    sklearn_metric: Callable,
    torch_metric: Metric,
    size: int,
    n_documents: int,
    query_without_relevant_docs_options: str,
    **kwargs,
) -> None:
    """ Compare PL metrics to standard version. """
    metric = torch_metric(query_without_relevant_docs=query_without_relevant_docs_options, **kwargs)
    shape = (size, )

    indexes = []
    preds = []
    target = []

    for i in range(n_documents):
        indexes.append(np.ones(shape, dtype=np.long) * i)
        preds.append(np.random.randn(*shape))
        target.append(np.random.randn(*shape) > 0)

    sk_results = _compute_sklearn_metric(sklearn_metric, target, preds, query_without_relevant_docs_options, **kwargs)
    sk_results = torch.tensor(sk_results)

    indexes_tensor = torch.cat([torch.tensor(i) for i in indexes]).long()
    preds_tensor = torch.cat([torch.tensor(p) for p in preds]).float()
    target_tensor = torch.cat([torch.tensor(t) for t in target]).long()

    # lets assume data are not ordered
    perm = torch.randperm(indexes_tensor.nelement())
    indexes_tensor = indexes_tensor.view(-1)[perm].view(indexes_tensor.size())
    preds_tensor = preds_tensor.view(-1)[perm].view(preds_tensor.size())
    target_tensor = target_tensor.view(-1)[perm].view(target_tensor.size())

    # shuffle ids to require also sorting of documents ability from the torch metric
    pl_result = metric(indexes_tensor, preds_tensor, target_tensor)

    assert torch.allclose(sk_results.float(), pl_result.float(), equal_nan=False), (
        f"Test failed comparing metric {sklearn_metric} with {torch_metric}: "
        f"{sk_results.float()} vs {pl_result.float()}. "
        f"indexes: {indexes}, preds: {preds}, target: {target}"
    )


def _test_dtypes(torchmetric) -> None:
    """Check PL metrics inputs are controlled correctly. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    length = 10  # not important in this test

    # check error when `query_without_relevant_docs='error'` is raised correctly
    indexes = torch.tensor([0] * length, device=device, dtype=torch.int64)
    preds = torch.rand(size=(length, ), device=device, dtype=torch.float32)
    target = torch.tensor([False] * length, device=device, dtype=torch.bool)

    metric = torchmetric(query_without_relevant_docs='error')
    with pytest.raises(ValueError, match="`compute` method was provided with a query with no positive target."):
        metric(indexes, preds, target)

    # check ValueError with invalid `query_without_relevant_docs` argument
    casual_argument = 'casual_argument'
    with pytest.raises(ValueError, match=f"`query_without_relevant_docs` received a wrong value {casual_argument}."):
        metric = torchmetric(query_without_relevant_docs=casual_argument)

    # check input dtypes
    indexes = torch.tensor([0] * length, device=device, dtype=torch.int64)
    preds = torch.tensor([0] * length, device=device, dtype=torch.float32)
    target = torch.tensor([0] * length, device=device, dtype=torch.int64)

    metric = torchmetric(query_without_relevant_docs='error')

    # check error on input dtypes are raised correctly
    with pytest.raises(ValueError, match="`indexes` must be a tensor of long integers"):
        metric(indexes.bool(), preds, target)
    with pytest.raises(ValueError, match="`preds` must be a tensor of floats"):
        metric(indexes, preds.bool(), target)
    with pytest.raises(ValueError, match="`target` must be a tensor of booleans or integers"):
        metric(indexes, preds, target.float())


def _test_input_shapes(torchmetric) -> None:
    """Check PL metrics inputs are controlled correctly. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric = torchmetric(query_without_relevant_docs='error')

    # check input shapes are checked correclty
    elements_1, elements_2 = np.random.choice(np.arange(1, 20), size=2, replace=False)
    indexes = torch.tensor([0] * elements_1, device=device, dtype=torch.int64)
    preds = torch.tensor([0] * elements_2, device=device, dtype=torch.float32)
    target = torch.tensor([0] * elements_2, device=device, dtype=torch.int64)

    with pytest.raises(ValueError, match="`indexes`, `preds` and `target` must be of the same shape"):
        metric(indexes, preds, target)


def _test_input_args(torchmetric: Metric, message: str, **kwargs) -> None:
    """Check invalid args are managed correctly. """
    with pytest.raises(ValueError, match=message):
        torchmetric(**kwargs)

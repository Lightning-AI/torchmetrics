import math
import random
from typing import Callable, List

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score as sk_average_precision
from torch import Tensor

from torchmetrics.metric import Metric
from torchmetrics.retrieval.mean_average_precision import RetrievalMAP


@pytest.mark.parametrize('sklearn_metric,torch_class_metric', [
    [sk_average_precision, RetrievalMAP],
])
@pytest.mark.parametrize("size", [1, 4, 10, 100])
@pytest.mark.parametrize("batch", [1, 4, 10])
@pytest.mark.parametrize("behaviour", ['skip', 'pos', 'neg'])
def test_against_sklearn(sklearn_metric: Callable, torch_class_metric: Metric, size, batch, behaviour) -> None:
    """Compare PL metrics to sklearn version. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    metric = torch_class_metric(query_without_relevant_docs=behaviour)
    shape = (size, )

    indexes = []
    preds = []
    target = []

    for i in range(batch):
        indexes.append(np.ones(shape, dtype=int) * i)
        preds.append(np.random.randn(*shape))
        target.append(np.random.randn(*shape) > 0)

    sk_results = _compute_sklearn_metric(target, preds, behaviour, sklearn_metric, device)

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

    assert torch.allclose(sk_results.float(), pl_result.float(), equal_nan=True)


def _compute_sklearn_metric(
    target: List[np.ndarray], preds: List[np.ndarray], behaviour: str, sklearn_metric: Callable, device
) -> Tensor:
    """ Compute sk metric with multiple iterations using the base `sklearn_metric`. """
    sk_results = []
    kwargs = {'device': device, 'dtype': torch.float32}

    for b, a in zip(target, preds):
        res = sklearn_metric(b, a)

        if math.isnan(res):
            if behaviour == 'skip':
                pass
            elif behaviour == 'pos':
                sk_results.append(torch.tensor(1.0, **kwargs))
            else:
                sk_results.append(torch.tensor(0.0, **kwargs))
        else:
            sk_results.append(torch.tensor(res, **kwargs))
    if len(sk_results) > 0:
        sk_results = torch.stack(sk_results).mean()
    else:
        sk_results = torch.tensor(0.0, **kwargs)

    return sk_results


@pytest.mark.parametrize(['torch_class_metric'], [
    [RetrievalMAP],
])
def test_input_data(torch_class_metric: Metric) -> None:
    """Check PL metrics inputs are controlled correctly. """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    length = random.randint(0, 20)

    # check error when `query_without_relevant_docs='error'` is raised correctly
    indexes = torch.tensor([0] * length, device=device, dtype=torch.int64)
    preds = torch.rand(size=(length, ), device=device, dtype=torch.float32)
    target = torch.tensor([False] * length, device=device, dtype=torch.bool)

    metric = torch_class_metric(query_without_relevant_docs='error')

    with pytest.raises(ValueError):
        metric(indexes, preds, target)

    # check ValueError with non-accepted argument
    with pytest.raises(ValueError):
        torch_class_metric(query_without_relevant_docs='casual_argument')

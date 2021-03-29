import math

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score as sk_average_precision

from tests.helpers import seed_all
from tests.retrieval.test_mrr import _reciprocal_rank as reciprocal_rank
from tests.retrieval.test_precision import _precision_at_k as precision_at_k
from tests.retrieval.test_recall import _recall_at_k as recall_at_k
from torchmetrics.functional.retrieval.average_precision import retrieval_average_precision
from torchmetrics.functional.retrieval.precision import retrieval_precision
from torchmetrics.functional.retrieval.recall import retrieval_recall
from torchmetrics.functional.retrieval.reciprocal_rank import retrieval_reciprocal_rank

seed_all(1337)


@pytest.mark.parametrize(['sklearn_metric', 'torch_metric'], [
    [sk_average_precision, retrieval_average_precision],
    [reciprocal_rank, retrieval_reciprocal_rank],
])
@pytest.mark.parametrize("size", [1, 4, 10])
def test_metrics_output_values(sklearn_metric, torch_metric, size):
    """ Compare PL metrics to sklearn version. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # test results are computed correctly wrt std implementation
    for i in range(6):
        preds = np.random.randn(size)
        target = np.random.randn(size) > 0

        # sometimes test with integer targets
        if (i % 2) == 0:
            target = target.astype(np.int)

        sk = torch.tensor(sklearn_metric(target, preds), device=device)
        tm = torch_metric(torch.tensor(preds, device=device), torch.tensor(target, device=device))

        # `torch_metric`s return 0 when no label is True
        # while `sklearn` metrics returns NaN
        if math.isnan(sk):
            assert tm == 0
        else:
            assert torch.allclose(sk.float(), tm.float())


@pytest.mark.parametrize(['sklearn_metric', 'torch_metric'], [
    [precision_at_k, retrieval_precision],
    [recall_at_k, retrieval_recall],
])
@pytest.mark.parametrize("size", [1, 4, 10])
@pytest.mark.parametrize("k", [None, 1, 4, 10])
def test_metrics_output_values_with_k(sklearn_metric, torch_metric, size, k):
    """ Compare PL metrics to sklearn version. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # test results are computed correctly wrt std implementation
    for i in range(6):
        preds = np.random.randn(size)
        target = np.random.randn(size) > 0

        # sometimes test with integer targets
        if (i % 2) == 0:
            target = target.astype(np.int)

        sk = torch.tensor(sklearn_metric(target, preds, k), device=device)
        tm = torch_metric(torch.tensor(preds, device=device), torch.tensor(target, device=device), k)

        # `torch_metric`s return 0 when no label is True
        # while `sklearn` metrics returns NaN
        if math.isnan(sk):
            assert tm == 0
        else:
            assert torch.allclose(sk.float(), tm.float())


@pytest.mark.parametrize(['torch_metric'], [
    [retrieval_average_precision],
    [retrieval_reciprocal_rank],
    [retrieval_precision],
    [retrieval_recall],
])
def test_input_dtypes(torch_metric) -> None:
    """ Check wrong input dtypes are managed correctly. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    length = 10  # not important in this case

    # check target is binary
    preds = torch.tensor([0.0, 1.0] * length, device=device, dtype=torch.float32)
    target = torch.tensor([-1, 2] * length, device=device, dtype=torch.int64)

    with pytest.raises(ValueError, match="`target` must be of type `binary`"):
        torch_metric(preds, target)

    # check dtypes and empty target
    preds = torch.tensor([0] * length, device=device, dtype=torch.float32)
    target = torch.tensor([0] * length, device=device, dtype=torch.int64)

    # check error on input dtypes are raised correctly
    with pytest.raises(ValueError, match="`preds` must be a tensor of floats"):
        torch_metric(preds.bool(), target)
    with pytest.raises(ValueError, match="`target` must be a tensor of booleans or integers"):
        torch_metric(preds, target.float())

    # test checks on empty targets
    assert torch.allclose(torch_metric(preds=preds, target=target), torch.tensor(0.0))


@pytest.mark.parametrize(['torch_metric'], [
    [retrieval_average_precision],
    [retrieval_reciprocal_rank],
    [retrieval_precision],
    [retrieval_recall],
])
def test_input_shapes(torch_metric) -> None:
    """ Check wrong input shapes are managed correctly. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # test with empty tensors
    preds = torch.tensor([0] * 0, device=device, dtype=torch.float)
    target = torch.tensor([0] * 0, device=device, dtype=torch.int64)
    with pytest.raises(ValueError, match="`preds` and `target` must be non-empty"):
        torch_metric(preds, target)

    # test checks when shapes are different
    elements_1, elements_2 = np.random.choice(np.arange(1, 20), size=2, replace=False)  # ensure sizes are different
    preds = torch.tensor([0] * elements_1, device=device, dtype=torch.float)
    target = torch.tensor([0] * elements_2, device=device, dtype=torch.int64)

    with pytest.raises(ValueError, match="`preds` and `target` must be of the same shape"):
        torch_metric(preds, target)


# test metrics using top K parameter
@pytest.mark.parametrize(['torch_metric'], [
    [retrieval_precision],
    [retrieval_recall],
])
@pytest.mark.parametrize('k', [-1, 1.0])
def test_input_params(torch_metric, k) -> None:
    """ Check wrong input shapes are managed correctly. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # test with random tensors
    preds = torch.tensor([0] * 4, device=device, dtype=torch.float)
    target = torch.tensor([0] * 4, device=device, dtype=torch.int64)
    with pytest.raises(ValueError, match="`k` has to be a positive integer or None"):
        torch_metric(preds, target, k=k)

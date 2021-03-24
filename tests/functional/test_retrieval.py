import math

import numpy as np
import pytest
import torch
from pytorch_lightning import seed_everything
from sklearn.metrics import average_precision_score as sk_average_precision

from tests.retrieval.helpers import _assert_error
from tests.retrieval.test_mrr import _reciprocal_rank as reciprocal_rank
from torchmetrics.functional.retrieval.average_precision import retrieval_average_precision
from torchmetrics.functional.retrieval.reciprocal_rank import retrieval_reciprocal_rank


@pytest.mark.parametrize(['sklearn_metric', 'torch_metric'], [
    [sk_average_precision, retrieval_average_precision],
    [reciprocal_rank, retrieval_reciprocal_rank],
])
@pytest.mark.parametrize("size", [1, 4, 10])
def test_metrics_output_values(sklearn_metric, torch_metric, size):
    """ Compare PL metrics to sklearn version. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(0)

    # test results are computed correctly wrt std implementation
    for i in range(6):
        preds = np.random.randn(size)
        target = np.random.randn(size) > 0

        # sometimes test with integer targets
        if (i % 2) == 0:
            target = target.astype(np.int)

        sk = torch.tensor(sklearn_metric(target, preds), device=device)
        pl = torch_metric(torch.tensor(preds, device=device), torch.tensor(target, device=device))

        # `torch_metric`s return 0 when no label is True
        # while `sklearn` metrics returns NaN
        if math.isnan(sk):
            assert pl == 0
        else:
            assert torch.allclose(sk.float(), pl.float())


@pytest.mark.parametrize(['torch_metric'], [
    [retrieval_average_precision],
    [retrieval_reciprocal_rank],
])
def test_input_dtypes(torch_metric) -> None:
    """ Check wrong input dtypes are managed correctly. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(0)

    length = 10  # not important in this case
    preds = torch.tensor([0] * length, device=device, dtype=torch.float32)
    target = torch.tensor([0] * length, device=device, dtype=torch.int64)

    # check error on input dtypes are raised correctly
    _assert_error(torch_metric, ValueError, preds.bool(), target)
    _assert_error(torch_metric, ValueError, preds, target.float())

    # test checks on empty targets
    assert torch.allclose(torch_metric(preds=preds, target=target), torch.tensor(0.0))


@pytest.mark.parametrize(['torch_metric'], [
    [retrieval_average_precision],
    [retrieval_reciprocal_rank],
])
def test_input_shapes(torch_metric) -> None:
    """ Check wrong input shapes are managed correctly. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(0)

    # test with empty tensors
    preds = torch.tensor([0] * 0, device=device, dtype=torch.float)
    target = torch.tensor([0] * 0, device=device, dtype=torch.int64)
    _assert_error(torch_metric, ValueError, preds, target)

    # test checks when shapes are different
    elements_1, elements_2 = np.random.choice(np.arange(1, 20), size=2, replace=False)  # ensure sizes are different
    preds = torch.tensor([0] * elements_1, device=device, dtype=torch.float)
    target = torch.tensor([0] * elements_2, device=device, dtype=torch.int64)
    _assert_error(torch_metric, ValueError, preds, target)

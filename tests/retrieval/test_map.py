import random
from typing import List

import numpy as np
import torch
from pytorch_lightning import seed_everything
from sklearn.metrics import average_precision_score as sk_average_precision
from torch import Tensor

from torchmetrics.retrieval.mean_average_precision import RetrievalMAP


def test_against_sklearn() -> None:
    """Compare PL metrics to sklearn version. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(0)

    rounds = 20
    sizes = [1, 4, 10, 100]
    batch_sizes = [1, 4, 10]
    query_without_relevant_docs_options = ['skip', 'pos', 'neg']

    def compute_sklearn_metric(target: List[np.ndarray], preds: List[np.ndarray], behaviour: str) -> Tensor:
        """ Compute sk metric with multiple iterations using the base `sk_average_precision`. """
        sk_results = []
        kwargs = {'device': device, 'dtype': torch.float32}

        for b, a in zip(target, preds):

            if b.sum() == 0:
                if behaviour == 'skip':
                    pass
                elif behaviour == 'pos':
                    sk_results.append(1.0)
                else:
                    sk_results.append(0.0)
            else:
                res = sk_average_precision(b, a)
                sk_results.append(res)

        sk_results = torch.tensor(sk_results, **kwargs)

        if sk_results.numel() > 0:
            sk_results = sk_results.mean()
        else:
            sk_results = torch.tensor(0.0, **kwargs)

        return sk_results

    def do_test(batch_size: int, size: int) -> None:
        """ For each possible behaviour of the metric, check results are correct. """
        for behaviour in query_without_relevant_docs_options:

            metric = RetrievalMAP(query_without_relevant_docs=behaviour)
            shape = (size, )

            indexes = []
            preds = []
            target = []

            for i in range(batch_size):
                indexes.append(np.ones(shape, dtype=int) * i)
                preds.append(np.random.randn(*shape))
                target.append(np.random.randn(*shape) > 0)

            sk_results = compute_sklearn_metric(target, preds, behaviour)

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

    for batch_size in batch_sizes:
        for size in sizes:
            for _ in range(rounds):
                do_test(batch_size, size)


def test_input_data() -> None:
    """Check PL metrics inputs are controlled correctly. """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(0)

    for _ in range(10):

        length = random.randint(0, 20)

        # check error when `query_without_relevant_docs='error'` is raised correctly
        indexes = torch.tensor([0] * length, device=device, dtype=torch.int64)
        preds = torch.rand(size=(length, ), device=device, dtype=torch.float32)
        target = torch.tensor([False] * length, device=device, dtype=torch.bool)

        metric = RetrievalMAP(query_without_relevant_docs='error')

        try:
            metric(indexes, preds, target)
        except Exception as e:
            assert isinstance(e, ValueError)

        # check ValueError with invalid `query_without_relevant_docs` argument
        try:
            metric = RetrievalMAP(query_without_relevant_docs='casual_argument')
        except Exception as e:
            assert isinstance(e, ValueError)

    # check input dtypes
    NOT_ALLOWED_INDEXES_DTYPE = (torch.bool, torch.float)
    NOT_ALLOWED_PREDS_DTYPE = (torch.bool, )
    NOW_ALLOWED_TARGET_DTYPE = (torch.float, )

    length = 10  # not important in this case
    for dtype in NOT_ALLOWED_INDEXES_DTYPE + NOT_ALLOWED_PREDS_DTYPE + NOW_ALLOWED_TARGET_DTYPE:

        # check error input dtypes error is raised correctly
        indexes = torch.tensor([0] * length, device=device, dtype=dtype)
        preds = torch.tensor([0] * length, device=device, dtype=dtype)
        target = torch.tensor([0] * length, device=device, dtype=dtype)

        metric = RetrievalMAP(query_without_relevant_docs='skip')

        try:
            metric(indexes, preds, target)
        except Exception as e:
            assert isinstance(e, ValueError)

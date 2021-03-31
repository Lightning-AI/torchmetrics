# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torchmetrics.utilities.data import get_group_indexes
from typing import Callable, Union

import numpy as np
import pytest
import torch
from torch import Tensor

from tests.helpers import seed_all
from torchmetrics import Metric
from tests.helpers.testers import MetricTester

seed_all(1337)


def _compute_sklearn_metric(
    preds: Union[Tensor, np.ndarray],
    target: Union[Tensor, np.ndarray],
    idx: np.ndarray = None,
    metric: Callable = None,
    empty_target_action: str = "skip",
    **kwargs
) -> Tensor:
    """ Compute metric with multiple iterations over every query predictions set. """

    if isinstance(preds, Tensor):
        preds = preds.cpu().numpy()
    if isinstance(target, Tensor):
        target = target.cpu().numpy()

    if idx is None:
        idx = np.full_like(preds, fill_value=0, dtype=np.int64)

    groups = get_group_indexes(idx)
    sk_results = []
    for group in groups:
        trg, pds = target[group], preds[group]

        if trg.sum() == 0:
            if empty_target_action == 'skip':
                pass
            elif empty_target_action == 'pos':
                sk_results.append(1.0)
            else:
                sk_results.append(0.0)
        else:
            res = metric(trg, pds, **kwargs)
            sk_results.append(res)

    if len(sk_results) > 0:
        return np.mean(sk_results)
    return np.array(0.0)


def _test_retrieval_against_sklearn(
    sklearn_metric: Callable,
    torch_metric: Metric,
    size: int,
    n_documents: int,
    empty_target_action: str,
    **kwargs
) -> None:
    """ Compare PL metrics to standard version. """
    metric = torch_metric(empty_target_action=empty_target_action, **kwargs)
    shape = (n_documents, size)

    indexes = np.ones(shape, dtype=np.int64) * np.arange(n_documents)
    preds = np.random.randn(*shape)
    target = np.random.randint(0, 2, size=shape)

    sk_results = _compute_sklearn_metric(
        preds, target, metric=sklearn_metric, empty_target_action=empty_target_action, **kwargs
    )
    sk_results = torch.tensor(sk_results)

    indexes_tensor = torch.tensor(indexes).long()
    preds_tensor = torch.tensor(preds).float()
    target_tensor = torch.tensor(target).long()

    # lets assume data are not ordered
    perm = torch.randperm(indexes_tensor.nelement())
    indexes_tensor = indexes_tensor.view(-1)[perm].view(indexes_tensor.size())
    preds_tensor = preds_tensor.view(-1)[perm].view(preds_tensor.size())
    target_tensor = target_tensor.view(-1)[perm].view(target_tensor.size())

    # shuffle ids to require also sorting of documents ability from the torch metric
    pl_result = metric(preds_tensor, target_tensor, idx=indexes_tensor)

    assert torch.allclose(sk_results.float(), pl_result.float(), equal_nan=False), (
        f"Test failed comparing metric {sklearn_metric} with {torch_metric}: "
        f"{sk_results.float()} vs {pl_result.float()}. "
        f"indexes: {indexes}, preds: {preds}, target: {target}"
    )


def _test_dtypes(torchmetric) -> None:
    """Check PL metrics inputs are controlled correctly. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    length = 10  # not important in this test

    # check error when `empty_target_action='error'` is raised correctly
    indexes = torch.tensor([0] * length, device=device, dtype=torch.int64)
    preds = torch.rand(size=(length, ), device=device, dtype=torch.float32)
    target = torch.tensor([False] * length, device=device, dtype=torch.bool)

    metric = torchmetric(empty_target_action='error')
    with pytest.raises(ValueError, match="`compute` method was provided with a query with no positive target."):
        metric(preds, target, idx=indexes)

    # check ValueError with invalid `empty_target_action` argument
    casual_argument = 'casual_argument'
    with pytest.raises(ValueError, match=f"`empty_target_action` received a wrong value {casual_argument}."):
        metric = torchmetric(empty_target_action=casual_argument)

    # check input dtypes
    indexes = torch.tensor([0] * length, device=device, dtype=torch.int64)
    preds = torch.tensor([0] * length, device=device, dtype=torch.float32)
    target = torch.tensor([0] * length, device=device, dtype=torch.int64)

    metric = torchmetric(empty_target_action='error')

    # check error on input dtypes are raised correctly
    with pytest.raises(ValueError, match="`indexes` must be a tensor of long integers"):
        metric(preds, target, idx=indexes.bool())
    with pytest.raises(ValueError, match="`preds` must be a tensor of floats"):
        metric(preds.bool(), target, idx=indexes)
    with pytest.raises(ValueError, match="`target` must be a tensor of booleans or integers"):
        metric(preds, target.float(), idx=indexes)


def _test_input_shapes(torchmetric) -> None:
    """Check PL metrics inputs are controlled correctly. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metric = torchmetric(empty_target_action='error')

    # check input shapes are checked correclty
    elements_1, elements_2 = np.random.choice(np.arange(1, 20), size=2, replace=False)
    indexes = torch.tensor([0] * elements_1, device=device, dtype=torch.int64)
    preds = torch.tensor([0] * elements_2, device=device, dtype=torch.float32)
    target = torch.tensor([0] * elements_2, device=device, dtype=torch.int64)

    with pytest.raises(ValueError, match="`indexes`, `preds` and `target` must be of the same shape"):
        metric(preds, target, idx=indexes)


def _test_input_args(torchmetric: Metric, message: str, **kwargs) -> None:
    """Check invalid args are managed correctly. """
    with pytest.raises(ValueError, match=message):
        torchmetric(**kwargs)







class RetrievalMetricTester(MetricTester):

    """
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_average_precision(self, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=RetrievalMAP,
            sk_metric=sk_metric,
            dist_sync_on_step=dist_sync_on_step,
        )
    """

    def test_average_precision_functional(self, preds, target, sk_metric):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=retrieval_average_precision,
            sk_metric=sk_metric,
        )

    def test_a_caso(self, preds, target, sk_metric):
        assert False
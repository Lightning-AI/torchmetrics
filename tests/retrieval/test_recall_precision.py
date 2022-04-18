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


from typing import Tuple, Union

import numpy as np
import pytest
from numpy import array
from torch import Tensor, tensor

from tests import MetricTester
from tests.helpers import seed_all
from tests.retrieval.helpers import _irs, get_group_indexes
from tests.retrieval.test_precision import _precision_at_k
from tests.retrieval.test_recall import _recall_at_k
from torchmetrics import RetrievalRecallAtFixedPrecision

seed_all(42)


def _compute_recall_at_precision_metric(
    preds: Union[Tensor, array],
    target: Union[Tensor, array],
    indexes: Union[Tensor, array] = None,
    max_k: int = None,
    min_precision: float = 0.0,
    ignore_index: int = None,
) -> Tuple[Tensor, int]:
    """Compute metric with multiple iterations over every query predictions set."""
    recalls, precisions = [], []

    if indexes is None:
        indexes = np.full_like(preds, fill_value=0, dtype=np.int64)
    if isinstance(indexes, Tensor):
        indexes = indexes.cpu().numpy()
    if isinstance(preds, Tensor):
        preds = preds.cpu().numpy()
    if isinstance(target, Tensor):
        target = target.cpu().numpy()

    assert isinstance(indexes, np.ndarray)
    assert isinstance(preds, np.ndarray)
    assert isinstance(target, np.ndarray)

    if ignore_index is not None:
        valid_positions = target != ignore_index
        indexes, preds, target = indexes[valid_positions], preds[valid_positions], target[valid_positions]

    indexes = indexes.flatten()
    preds = preds.flatten()
    target = target.flatten()
    groups = get_group_indexes(indexes)

    if max_k is None:
        max_k = max(map(len, groups))

    max_k_range = list(range(1, max_k + 1))

    for group in groups:
        trg, prd = target[group], preds[group]
        r, p = [], []

        for k in max_k_range:
            r.append(_recall_at_k(trg, prd, k=k))
            p.append(_precision_at_k(trg, prd, k=k))

        recalls.append(r)
        precisions.append(p)

    recalls = tensor(recalls).mean(dim=0)
    precisions = tensor(precisions).mean(dim=0)

    recalls_at_k = [(r, k) for p, r, k in zip(precisions, recalls, max_k_range) if p >= min_precision]

    assert recalls_at_k

    return max(recalls_at_k)


def test_compute_recall_at_precision_metric():
    indexes = tensor([0, 0, 0, 0, 1, 1, 1])
    preds = tensor([0.4, 0.01, 0.5, 0.6, 0.2, 0.3, 0.5])
    target = tensor([True, False, False, True, True, False, True])
    max_k = 3
    min_precision = 0.8

    res = _compute_recall_at_precision_metric(
        preds,
        target,
        indexes,
        max_k,
        min_precision,
    )
    assert res == (tensor(0.5000), 1)


@pytest.mark.parametrize("indexes,preds,target", [(i, p, t) for i, p, t in zip(_irs.indexes, _irs.preds, _irs.target)])
@pytest.mark.parametrize("ddp", [False])
@pytest.mark.parametrize("dist_sync_on_step", [False])
@pytest.mark.parametrize("empty_target_action", ["skip", "neg", "pos"])
@pytest.mark.parametrize("ignore_index", [None, 1])  # avoid setting 0, otherwise test with all 0 targets will fail
@pytest.mark.parametrize("max_k", [None, 1, 4, 10])
@pytest.mark.parametrize("min_precision", [0.0, 0.2])
class TestRetrievalRecallAtFixedPrecision(MetricTester):
    atol = 0.02

    def test_12312312(
        self, indexes, preds, target, ddp, dist_sync_on_step, empty_target_action, ignore_index, max_k, min_precision
    ):

        self.run_class_metric_test(
            ddp=ddp,
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalRecallAtFixedPrecision,
            sk_metric=_compute_recall_at_precision_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "max_k": max_k,
                "min_precision": min_precision,
                "ignore_index": ignore_index,
                "empty_target_action": empty_target_action,
            },
        )

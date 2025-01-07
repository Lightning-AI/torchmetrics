# Copyright The Lightning team.
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
from functools import partial
from typing import Callable, Optional, Union

import numpy as np
import pytest
import torch
from numpy import array
from torch import Tensor, tensor
from typing_extensions import Literal

from torchmetrics.retrieval import RetrievalPrecisionRecallCurve
from torchmetrics.retrieval.base import _retrieval_aggregate
from unittests._helpers import seed_all
from unittests._helpers.testers import Metric, MetricTester
from unittests.retrieval.helpers import _custom_aggregate_fn, _default_metric_class_input_arguments, get_group_indexes
from unittests.retrieval.test_precision import _precision_at_k
from unittests.retrieval.test_recall import _recall_at_k

seed_all(42)


def _compute_precision_recall_curve(
    preds: Union[Tensor, array],
    target: Union[Tensor, array],
    indexes: Optional[Union[Tensor, array]] = None,
    max_k: Optional[int] = None,
    adaptive_k: bool = False,
    ignore_index: Optional[int] = None,
    empty_target_action: str = "skip",
    reverse: bool = False,
    aggregation: Union[Literal["mean", "median", "min", "max"], Callable] = "mean",
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute metric with multiple iterations over every query predictions set.

    Didn't find a reliable implementation of precision-recall curve in Information Retrieval,
    so, reimplementing here.

    A good explanation can be found here:
    `<https://nlp.stanford.edu/IR-book/pdf/08eval.pdf>_`. (part 8.4)

    """
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

    top_k = torch.arange(1, max_k + 1)

    for group in groups:
        trg, prd = target[group], preds[group]
        r, p = [], []

        if ((1 - trg) if reverse else trg).sum() == 0:
            if empty_target_action == "skip":
                pass
            elif empty_target_action == "pos":
                arr = [1.0] * max_k
                recalls.append(arr)
                precisions.append(arr)
            elif empty_target_action == "neg":
                arr = [0.0] * max_k
                recalls.append(arr)
                precisions.append(arr)

        else:
            for k in top_k:
                r.append(_recall_at_k(trg, prd, top_k=k.item()))
                p.append(_precision_at_k(trg, prd, top_k=k.item(), adaptive_k=adaptive_k))

            recalls.append(r)
            precisions.append(p)

    if not recalls:
        return torch.zeros(max_k), torch.zeros(max_k), top_k

    recalls = _retrieval_aggregate(tensor(recalls), aggregation=aggregation, dim=0)
    precisions = _retrieval_aggregate(tensor(precisions), aggregation=aggregation, dim=0)

    return precisions, recalls, top_k


class RetrievalPrecisionRecallCurveTester(MetricTester):
    """Tester class for `RetrievalPrecisionRecallCurveTester` metric."""

    def run_class_metric_test(
        self,
        ddp: bool,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        metric_class: Metric,
        reference_metric: Callable,
        metric_args: dict,
        reverse: bool = False,
        aggregation: Union[Literal["mean", "median", "min", "max"], Callable] = "mean",
    ):
        """Test class implementation of metric."""
        _ref_metric_adapted = partial(reference_metric, reverse=reverse, **metric_args)

        super().run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            reference_metric=_ref_metric_adapted,
            metric_args=metric_args,
            fragment_kwargs=True,
            indexes=indexes,  # every additional argument will be passed to metric_class and _ref_metric_adapted
        )


@pytest.mark.parametrize("ddp", [False])
@pytest.mark.parametrize("empty_target_action", ["neg", "skip", "pos"])
@pytest.mark.parametrize("ignore_index", [None, 1])  # avoid setting 0, otherwise test with all 0 targets will fail
@pytest.mark.parametrize("max_k", [None, 1, 2, 5, 10])
@pytest.mark.parametrize("adaptive_k", [False, True])
@pytest.mark.parametrize("aggregation", ["mean", "median", "max", "min", _custom_aggregate_fn])
@pytest.mark.parametrize(**_default_metric_class_input_arguments)
class TestRetrievalPrecisionRecallCurve(RetrievalPrecisionRecallCurveTester):
    """Test class for `RetrievalPrecisionRecallCurveTester` metric."""

    atol = 0.02

    def test_class_metric(
        self,
        indexes,
        preds,
        target,
        ddp,
        empty_target_action,
        ignore_index,
        max_k,
        adaptive_k,
        aggregation,
    ):
        """Test class implementation of metric."""
        metric_args = {
            "max_k": max_k,
            "adaptive_k": adaptive_k,
            "empty_target_action": empty_target_action,
            "ignore_index": ignore_index,
            "aggregation": aggregation,
        }

        self.run_class_metric_test(
            ddp=ddp,
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalPrecisionRecallCurve,
            reference_metric=_compute_precision_recall_curve,
            metric_args=metric_args,
        )

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
from typing import Callable, Optional, Union

import numpy as np
import pytest
import torch
from sklearn.metrics import ndcg_score
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.retrieval.ndcg import retrieval_normalized_dcg
from torchmetrics.retrieval.ndcg import RetrievalNormalizedDCG
from unittests._helpers import seed_all
from unittests.retrieval.helpers import (
    RetrievalMetricTester,
    _concat_tests,
    _custom_aggregate_fn,
    _default_metric_class_input_arguments_ignore_index,
    _default_metric_class_input_arguments_with_non_binary_target,
    _default_metric_functional_input_arguments_with_non_binary_target,
    _errors_test_class_metric_parameters_k,
    _errors_test_class_metric_parameters_with_nonbinary,
    _errors_test_functional_metric_parameters_k,
    _errors_test_functional_metric_parameters_with_nonbinary,
)

seed_all(42)


def _ndcg_at_k(target: np.ndarray, preds: np.ndarray, top_k: Optional[int] = None):
    """Adapting `from sklearn.metrics.ndcg_score`."""
    assert target.shape == preds.shape
    assert len(target.shape) == 1  # works only with single dimension inputs

    if target.shape[0] < 2:  # ranking is equal to ideal ranking with a single document
        return np.array(1.0)

    preds = np.expand_dims(preds, axis=0)
    target = np.expand_dims(target, axis=0)

    return ndcg_score(target, preds, k=top_k)


class TestNDCG(RetrievalMetricTester):
    """Test class for `RetrievalNormalizedDCG` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("empty_target_action", ["skip", "neg", "pos"])
    @pytest.mark.parametrize("ignore_index", [None, 3])  # avoid setting 0, otherwise test with all 0 targets will fail
    @pytest.mark.parametrize("k", [None, 1, 4, 10])
    @pytest.mark.parametrize("aggregation", ["mean", "median", "max", "min", _custom_aggregate_fn])
    @pytest.mark.parametrize(**_default_metric_class_input_arguments_with_non_binary_target)
    def test_class_metric(
        self,
        ddp: bool,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        empty_target_action: str,
        ignore_index: int,
        k: int,
        aggregation: Union[Literal["mean", "median", "min", "max"], Callable],
    ):
        """Test class implementation of metric."""
        metric_args = {
            "empty_target_action": empty_target_action,
            "top_k": k,
            "ignore_index": ignore_index,
            "aggregation": aggregation,
        }
        target = target if target.min() >= 0 else target - target.min()

        self.run_class_metric_test(
            ddp=ddp,
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalNormalizedDCG,
            reference_metric=_ndcg_at_k,
            metric_args=metric_args,
        )

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("empty_target_action", ["skip", "neg", "pos"])
    @pytest.mark.parametrize("k", [None, 1, 4, 10])
    @pytest.mark.parametrize(**_default_metric_class_input_arguments_ignore_index)
    def test_class_metric_ignore_index(
        self,
        ddp: bool,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        empty_target_action: str,
        k: int,
    ):
        """Test class implementation of metric with ignore_index argument."""
        metric_args = {"empty_target_action": empty_target_action, "top_k": k, "ignore_index": -100}

        target = target if target.min() >= 0 else target - target.min()
        self.run_class_metric_test(
            ddp=ddp,
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalNormalizedDCG,
            reference_metric=_ndcg_at_k,
            metric_args=metric_args,
        )

    @pytest.mark.parametrize(**_default_metric_functional_input_arguments_with_non_binary_target)
    @pytest.mark.parametrize("k", [None, 1, 4, 10])
    def test_functional_metric(self, preds: Tensor, target: Tensor, k: int):
        """Test functional implementation of metric."""
        target = target if target.min() >= 0 else target - target.min()
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_normalized_dcg,
            reference_metric=_ndcg_at_k,
            metric_args={},
            top_k=k,
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments_with_non_binary_target)
    def test_precision_cpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
        """Test dtype support of the metric on CPU."""
        target = target if target.min() >= 0 else target - target.min()
        self.run_precision_test_cpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalNormalizedDCG,
            metric_functional=retrieval_normalized_dcg,
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments_with_non_binary_target)
    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
        """Test dtype support of the metric on GPU."""
        target = target if target.min() >= 0 else target - target.min()
        self.run_precision_test_gpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalNormalizedDCG,
            metric_functional=retrieval_normalized_dcg,
        )

    @pytest.mark.parametrize(
        **_concat_tests(
            _errors_test_class_metric_parameters_with_nonbinary,
            _errors_test_class_metric_parameters_k,
        )
    )
    def test_arguments_class_metric(
        self, indexes: Tensor, preds: Tensor, target: Tensor, message: str, metric_args: dict
    ):
        """Test that specific errors are raised for incorrect input."""
        if target.is_floating_point():
            pytest.skip("NDCG metric works with float target input")

        self.run_metric_class_arguments_test(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalNormalizedDCG,
            message=message,
            metric_args=metric_args,
            exception_type=ValueError,
            kwargs_update={},
        )

    @pytest.mark.parametrize(
        **_concat_tests(
            _errors_test_functional_metric_parameters_with_nonbinary,
            _errors_test_functional_metric_parameters_k,
        )
    )
    def test_arguments_functional_metric(self, preds: Tensor, target: Tensor, message: str, metric_args: dict):
        """Test that specific errors are raised for incorrect input."""
        if target.is_floating_point():
            pytest.skip("NDCG metric works with float target input")

        self.run_functional_metric_arguments_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_normalized_dcg,
            message=message,
            exception_type=ValueError,
            kwargs_update=metric_args,
        )


def test_corner_case_with_tied_scores():
    """See issue: https://github.com/Lightning-AI/torchmetrics/issues/2022."""
    target = torch.tensor([[10, 0, 0, 1, 5]])
    preds = torch.tensor([[0.1, 0, 0, 0, 0.1]])

    for k in [1, 3, 5]:
        assert torch.allclose(
            retrieval_normalized_dcg(preds, target, top_k=k),
            torch.tensor([ndcg_score(target, preds, k=k)], dtype=torch.float32),
        )

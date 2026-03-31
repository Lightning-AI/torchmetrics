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


# ---- Tests for vectorized GPU-efficient implementation (issue #2287) ----


@pytest.mark.parametrize(
    ("batch_size", "list_length", "top_k"),
    [
        (1, 50, None),
        (1, 100, 10),
        (8, 50, None),
        (8, 100, 50),
        (32, 100, None),
        (32, 500, 200),
        (128, 100, 10),
        (128, 500, None),
    ],
)
def test_accuracy_vs_sklearn(batch_size: int, list_length: int, top_k: Optional[int]):
    """Batched nDCG must stay within 1e-4 of sklearn across configs.

    See issue: https://github.com/Lightning-AI/torchmetrics/issues/2287.
    """
    torch.manual_seed(42)
    scores = torch.randn(batch_size, list_length)
    labels = (torch.randint(0, 2, (batch_size, list_length)) * 2 - 1).float() + 1.0

    fast_result = retrieval_normalized_dcg(scores, labels, top_k=top_k).item()
    sklearn_result = float(np.mean([ndcg_score([t], [p], k=top_k) for t, p in zip(labels.numpy(), scores.numpy())]))

    assert abs(fast_result - sklearn_result) <= 1e-4, (
        f"nDCG differs from sklearn by {abs(fast_result - sklearn_result):.2e} "
        f"(B={batch_size}, L={list_length}, k={top_k})"
    )


def test_batched_input_matches_per_query():
    """Batched 2-D input must give the same mean nDCG as averaging per-query 1-D results.

    See issue: https://github.com/Lightning-AI/torchmetrics/issues/2287.
    """
    torch.manual_seed(42)
    preds = torch.randn(16, 50)
    target = (torch.randint(0, 2, (16, 50)) * 2 - 1).float() + 1.0

    per_query = torch.stack([retrieval_normalized_dcg(preds[i], target[i]) for i in range(preds.shape[0])])
    batched = retrieval_normalized_dcg(preds, target)

    assert torch.allclose(batched, per_query.mean(), atol=1e-5)


def test_tie_handling_explicit():
    """Tie-averaged DCG must match sklearn on inputs with explicit score ties.

    See issue: https://github.com/Lightning-AI/torchmetrics/issues/2287.
    """
    scores = torch.tensor([
        [1.0, 1.0, 0.5, 0.5, 0.1],  # two pairs of ties
        [0.8, 0.8, 0.8, 0.2, 0.1],  # three-way tie
    ])
    labels = torch.tensor([
        [1.0, 0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0, 0.0],
    ])

    result = retrieval_normalized_dcg(scores, labels)
    sklearn_result = float(np.mean([ndcg_score([t], [p]) for t, p in zip(labels.numpy(), scores.numpy())]))

    assert isinstance(result, torch.Tensor)
    assert 0.0 <= result.item() <= 1.0
    assert abs(result.item() - sklearn_result) <= 1e-4


def test_all_zeros_target():
    """All-irrelevant queries (target all zero) must return 0, not NaN."""
    scores = torch.randn(4, 20)
    labels = torch.zeros(4, 20)
    result = retrieval_normalized_dcg(scores, labels)
    assert result.item() == 0.0


def test_perfect_ranking():
    """A perfectly-ranked list must return nDCG == 1.0."""
    labels = torch.tensor([[3.0, 2.0, 1.0, 0.0, 0.0]] * 4)
    scores = labels.clone()  # predictions match ideal order
    result = retrieval_normalized_dcg(scores, labels)
    assert torch.allclose(result, torch.tensor(1.0), atol=1e-5)


@pytest.mark.parametrize("top_k", [1, 10, 50, None])
def test_top_k_valid_range(top_k: Optional[int]):
    """Results must be in [0, 1] for all top_k values."""
    torch.manual_seed(0)
    scores = torch.randn(8, 100)
    labels = torch.randint(0, 3, (8, 100)).float()
    result = retrieval_normalized_dcg(scores, labels, top_k=top_k)
    assert 0.0 <= result.item() <= 1.0

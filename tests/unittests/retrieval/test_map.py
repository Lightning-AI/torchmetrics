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
from sklearn.metrics import average_precision_score as sk_average_precision_score
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.retrieval.average_precision import retrieval_average_precision
from torchmetrics.retrieval.average_precision import RetrievalMAP
from unittests._helpers import seed_all
from unittests.retrieval.helpers import (
    RetrievalMetricTester,
    _concat_tests,
    _custom_aggregate_fn,
    _default_metric_class_input_arguments,
    _default_metric_class_input_arguments_ignore_index,
    _default_metric_functional_input_arguments,
    _errors_test_class_metric_parameters_default,
    _errors_test_class_metric_parameters_no_pos_target,
    _errors_test_functional_metric_parameters_default,
)

seed_all(42)


def _average_precision_at_k(target: np.ndarray, preds: np.ndarray, top_k: Optional[int] = None):
    """Wrap reference metric to account for top_k argument."""
    assert target.shape == preds.shape
    assert len(target.shape) == 1
    top_k = top_k or len(preds)
    idx = np.argsort(preds, axis=0)[::-1][:top_k]
    target, preds = target[idx], preds[idx]
    return sk_average_precision_score(target, preds)


class TestMAP(RetrievalMetricTester):
    """Test class for `RetrievalMAP` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("empty_target_action", ["skip", "neg", "pos"])
    @pytest.mark.parametrize("ignore_index", [None, 1])  # avoid setting 0, otherwise test with all 0 targets will fail
    @pytest.mark.parametrize("top_k", [None, 1, 4, 10])
    @pytest.mark.parametrize("aggregation", ["mean", "median", "max", "min", _custom_aggregate_fn])
    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_class_metric(
        self,
        ddp: bool,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        empty_target_action: str,
        ignore_index: int,
        top_k: int,
        aggregation: Union[Literal["mean", "median", "min", "max"], Callable],
    ):
        """Test class implementation of metric."""
        metric_args = {
            "empty_target_action": empty_target_action,
            "ignore_index": ignore_index,
            "top_k": top_k,
            "aggregation": aggregation,
        }

        self.run_class_metric_test(
            ddp=ddp,
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalMAP,
            reference_metric=_average_precision_at_k,
            metric_args=metric_args,
        )

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("empty_target_action", ["skip", "neg", "pos"])
    @pytest.mark.parametrize("top_k", [None, 1, 4, 10])
    @pytest.mark.parametrize(**_default_metric_class_input_arguments_ignore_index)
    def test_class_metric_ignore_index(
        self,
        ddp: bool,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        empty_target_action: str,
        top_k: int,
    ):
        """Test class implementation of metric with ignore_index argument."""
        metric_args = {"empty_target_action": empty_target_action, "ignore_index": -100, "top_k": top_k}

        self.run_class_metric_test(
            ddp=ddp,
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalMAP,
            reference_metric=_average_precision_at_k,
            metric_args=metric_args,
        )

    @pytest.mark.parametrize(**_default_metric_functional_input_arguments)
    @pytest.mark.parametrize("top_k", [None, 1, 4, 10])
    def test_functional_metric(self, preds: Tensor, target: Tensor, top_k: int):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_average_precision,
            reference_metric=_average_precision_at_k,
            metric_args={},
            top_k=top_k,
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_precision_cpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalMAP,
            metric_functional=retrieval_average_precision,
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalMAP,
            metric_functional=retrieval_average_precision,
        )

    @pytest.mark.parametrize(
        **_concat_tests(
            _errors_test_class_metric_parameters_default,
            _errors_test_class_metric_parameters_no_pos_target,
        )
    )
    def test_arguments_class_metric(
        self, indexes: Tensor, preds: Tensor, target: Tensor, message: str, metric_args: dict
    ):
        """Test that specific errors are raised for incorrect input."""
        self.run_metric_class_arguments_test(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalMAP,
            message=message,
            metric_args=metric_args,
            exception_type=ValueError,
            kwargs_update={},
        )

    @pytest.mark.parametrize(**_errors_test_functional_metric_parameters_default)
    def test_arguments_functional_metric(self, preds: Tensor, target: Tensor, message: str, metric_args: dict):
        """Test that specific errors are raised for incorrect input."""
        self.run_functional_metric_arguments_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_average_precision,
            message=message,
            exception_type=ValueError,
            kwargs_update=metric_args,
        )

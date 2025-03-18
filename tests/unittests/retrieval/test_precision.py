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
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.retrieval.precision import retrieval_precision
from torchmetrics.retrieval.precision import RetrievalPrecision
from unittests._helpers import seed_all
from unittests.retrieval.helpers import (
    RetrievalMetricTester,
    _concat_tests,
    _custom_aggregate_fn,
    _default_metric_class_input_arguments,
    _default_metric_class_input_arguments_ignore_index,
    _default_metric_functional_input_arguments,
    _errors_test_class_metric_parameters_adaptive_k,
    _errors_test_class_metric_parameters_default,
    _errors_test_class_metric_parameters_k,
    _errors_test_class_metric_parameters_no_pos_target,
    _errors_test_functional_metric_parameters_adaptive_k,
    _errors_test_functional_metric_parameters_default,
    _errors_test_functional_metric_parameters_k,
)

seed_all(42)


def _precision_at_k(target: np.ndarray, preds: np.ndarray, top_k: Optional[int] = None, adaptive_k: bool = False):
    """Didn't find a reliable implementation of Precision in Information Retrieval, so, reimplementing here.

    A good explanation can be found
    `here <https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf>_`.

    """
    assert target.shape == preds.shape
    assert len(target.shape) == 1  # works only with single dimension inputs

    if top_k is None or (adaptive_k and top_k > len(preds)):
        top_k = len(preds)

    if target.sum() > 0:
        order_indexes = np.argsort(preds, axis=0)[::-1]
        relevant = np.sum(target[order_indexes][:top_k])
        return relevant * 1.0 / top_k
    return np.NaN


class TestPrecision(RetrievalMetricTester):
    """Test class for `RetrievalPrecision` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("empty_target_action", ["skip", "neg", "pos"])
    @pytest.mark.parametrize("ignore_index", [None, 1])  # avoid setting 0, otherwise test with all 0 targets will fail
    @pytest.mark.parametrize("k", [None, 1, 4, 10])
    @pytest.mark.parametrize("adaptive_k", [False, True])
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
        k: int,
        adaptive_k: bool,
        aggregation: Union[Literal["mean", "median", "min", "max"], Callable],
    ):
        """Test class implementation of metric."""
        metric_args = {
            "empty_target_action": empty_target_action,
            "top_k": k,
            "ignore_index": ignore_index,
            "adaptive_k": adaptive_k,
            "aggregation": aggregation,
        }

        self.run_class_metric_test(
            ddp=ddp,
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalPrecision,
            reference_metric=_precision_at_k,
            metric_args=metric_args,
        )

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("empty_target_action", ["skip", "neg", "pos"])
    @pytest.mark.parametrize("k", [None, 1, 4, 10])
    @pytest.mark.parametrize("adaptive_k", [False, True])
    @pytest.mark.parametrize(**_default_metric_class_input_arguments_ignore_index)
    def test_class_metric_ignore_index(
        self,
        ddp: bool,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        empty_target_action: str,
        k: int,
        adaptive_k: bool,
    ):
        """Test class implementation of metric with ignore_index argument."""
        metric_args = {
            "empty_target_action": empty_target_action,
            "top_k": k,
            "ignore_index": -100,
            "adaptive_k": adaptive_k,
        }

        self.run_class_metric_test(
            ddp=ddp,
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalPrecision,
            reference_metric=_precision_at_k,
            metric_args=metric_args,
        )

    @pytest.mark.parametrize(**_default_metric_functional_input_arguments)
    @pytest.mark.parametrize("k", [None, 1, 4, 10])
    @pytest.mark.parametrize("adaptive_k", [False, True])
    def test_functional_metric(self, preds: Tensor, target: Tensor, k: int, adaptive_k: bool):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_precision,
            reference_metric=_precision_at_k,
            metric_args={},
            top_k=k,
            adaptive_k=adaptive_k,
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_precision_cpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalPrecision,
            metric_functional=retrieval_precision,
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalPrecision,
            metric_functional=retrieval_precision,
        )

    @pytest.mark.parametrize(
        **_concat_tests(
            _errors_test_class_metric_parameters_default,
            _errors_test_class_metric_parameters_no_pos_target,
            _errors_test_class_metric_parameters_k,
            _errors_test_class_metric_parameters_adaptive_k,
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
            metric_class=RetrievalPrecision,
            message=message,
            metric_args=metric_args,
            exception_type=ValueError,
            kwargs_update={},
        )

    @pytest.mark.parametrize(
        **_concat_tests(
            _errors_test_functional_metric_parameters_default,
            _errors_test_functional_metric_parameters_k,
            _errors_test_functional_metric_parameters_adaptive_k,
        )
    )
    def test_arguments_functional_metric(self, preds: Tensor, target: Tensor, message: str, metric_args: dict):
        """Test that specific errors are raised for incorrect input."""
        self.run_functional_metric_arguments_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_precision,
            message=message,
            exception_type=ValueError,
            kwargs_update=metric_args,
        )

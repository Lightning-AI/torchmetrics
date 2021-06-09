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
import numpy as np
import pytest
from torch import Tensor

from tests.helpers import seed_all
from tests.retrieval.helpers import (
    RetrievalMetricTester,
    _concat_tests,
    _default_metric_class_input_arguments,
    _default_metric_functional_input_arguments,
    _errors_test_class_metric_parameters_default,
    _errors_test_class_metric_parameters_k,
    _errors_test_class_metric_parameters_no_pos_target,
    _errors_test_functional_metric_parameters_default,
    _errors_test_functional_metric_parameters_k,
)
from torchmetrics.functional.retrieval.recall import retrieval_recall
from torchmetrics.retrieval.retrieval_recall import RetrievalRecall

seed_all(42)


def _recall_at_k(target: np.ndarray, preds: np.ndarray, k: int = None):
    """
    Didn't find a reliable implementation of Recall in Information Retrieval, so,
    reimplementing here. See wikipedia for more information about definition.
    """
    assert target.shape == preds.shape
    assert len(target.shape) == 1  # works only with single dimension inputs

    if k is None:
        k = len(preds)

    if target.sum() > 0:
        order_indexes = np.argsort(preds, axis=0)[::-1]
        relevant = np.sum(target[order_indexes][:k])
        return relevant * 1.0 / target.sum()
    return np.NaN


class TestRecall(RetrievalMetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    @pytest.mark.parametrize("empty_target_action", ['skip', 'neg', 'pos'])
    @pytest.mark.parametrize("k", [None, 1, 4, 10])
    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_class_metric(
        self,
        ddp: bool,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        dist_sync_on_step: bool,
        empty_target_action: str,
        k: int,
    ):
        metric_args = {'empty_target_action': empty_target_action, 'k': k}

        self.run_class_metric_test(
            ddp=ddp,
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalRecall,
            sk_metric=_recall_at_k,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
        )

    @pytest.mark.parametrize(**_default_metric_functional_input_arguments)
    @pytest.mark.parametrize("k", [None, 1, 4, 10])
    def test_functional_metric(self, preds: Tensor, target: Tensor, k: int):
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_recall,
            sk_metric=_recall_at_k,
            metric_args={},
            k=k,
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_precision_cpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
        self.run_precision_test_cpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalRecall,
            metric_functional=retrieval_recall,
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
        self.run_precision_test_gpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalRecall,
            metric_functional=retrieval_recall,
        )

    @pytest.mark.parametrize(
        **_concat_tests(
            _errors_test_class_metric_parameters_default,
            _errors_test_class_metric_parameters_no_pos_target,
            _errors_test_class_metric_parameters_k,
        )
    )
    def test_arguments_class_metric(
        self, indexes: Tensor, preds: Tensor, target: Tensor, message: str, metric_args: dict
    ):
        self.run_metric_class_arguments_test(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalRecall,
            message=message,
            metric_args=metric_args,
            exception_type=ValueError,
            kwargs_update={},
        )

    @pytest.mark.parametrize(
        **_concat_tests(
            _errors_test_functional_metric_parameters_default,
            _errors_test_functional_metric_parameters_k,
        )
    )
    def test_arguments_functional_metric(self, preds: Tensor, target: Tensor, message: str, metric_args: dict):
        self.run_functional_metric_arguments_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_recall,
            message=message,
            exception_type=ValueError,
            kwargs_update=metric_args,
        )

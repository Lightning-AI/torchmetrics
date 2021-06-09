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
from sklearn.metrics import label_ranking_average_precision_score
from torch import Tensor

from tests.helpers import seed_all
from tests.retrieval.helpers import (
    RetrievalMetricTester,
    _concat_tests,
    _default_metric_class_input_arguments,
    _default_metric_functional_input_arguments,
    _errors_test_class_metric_parameters_default,
    _errors_test_class_metric_parameters_no_pos_target,
    _errors_test_functional_metric_parameters_default,
)
from torchmetrics.functional.retrieval.reciprocal_rank import retrieval_reciprocal_rank
from torchmetrics.retrieval.mean_reciprocal_rank import RetrievalMRR

seed_all(42)


def _reciprocal_rank(target: np.ndarray, preds: np.ndarray):
    """
    Adaptation of `sklearn.metrics.label_ranking_average_precision_score`.
    Since the original sklearn metric works as RR only when the number of positive
    targets is exactly 1, here we remove every positive target that is not the most
    important. Remember that in RR only the positive target with the highest score is considered.
    """
    assert target.shape == preds.shape
    assert len(target.shape) == 1  # works only with single dimension inputs

    # going to remove T targets that are not ranked as highest
    indexes = preds[target.astype(bool)]
    if len(indexes) > 0:
        target[preds != indexes.max(-1, keepdims=True)[0]] = 0  # ensure that only 1 positive label is present

    if target.sum() > 0:
        # sklearn `label_ranking_average_precision_score` requires at most 2 dims
        return label_ranking_average_precision_score(np.expand_dims(target, axis=0), np.expand_dims(preds, axis=0))
    return 0.0


class TestMRR(RetrievalMetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    @pytest.mark.parametrize("empty_target_action", ['skip', 'neg', 'pos'])
    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_class_metric(
        self,
        ddp: bool,
        indexes: Tensor,
        preds: Tensor,
        target: Tensor,
        dist_sync_on_step: bool,
        empty_target_action: str,
    ):
        metric_args = {'empty_target_action': empty_target_action}

        self.run_class_metric_test(
            ddp=ddp,
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalMRR,
            sk_metric=_reciprocal_rank,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
        )

    @pytest.mark.parametrize(**_default_metric_functional_input_arguments)
    def test_functional_metric(self, preds: Tensor, target: Tensor):
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_reciprocal_rank,
            sk_metric=_reciprocal_rank,
            metric_args={},
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_precision_cpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
        self.run_precision_test_cpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalMRR,
            metric_functional=retrieval_reciprocal_rank,
        )

    @pytest.mark.parametrize(**_default_metric_class_input_arguments)
    def test_precision_gpu(self, indexes: Tensor, preds: Tensor, target: Tensor):
        self.run_precision_test_gpu(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_module=RetrievalMRR,
            metric_functional=retrieval_reciprocal_rank,
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
        self.run_metric_class_arguments_test(
            indexes=indexes,
            preds=preds,
            target=target,
            metric_class=RetrievalMRR,
            message=message,
            metric_args=metric_args,
            exception_type=ValueError,
            kwargs_update={},
        )

    @pytest.mark.parametrize(**_errors_test_functional_metric_parameters_default)
    def test_arguments_functional_metric(self, preds: Tensor, target: Tensor, message: str, metric_args: dict):
        self.run_functional_metric_arguments_test(
            preds=preds,
            target=target,
            metric_functional=retrieval_reciprocal_rank,
            message=message,
            exception_type=ValueError,
            kwargs_update=metric_args,
        )

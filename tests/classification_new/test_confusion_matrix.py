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

import pytest

from tests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester


@pytest.mark.parametrize(
    "preds, target",
    [
        (_input_binary_int.preds, _input_binary_int.target),
        (_input_binary_prob.preds, _input_binary_prob.target),
        (_input_binary_logit.preds, _input_binary_logit.target),
        (_input_binary_int_multidim.preds, _input_binary_int_multidim.target),
        (_input_binary_prob_multidim.preds, _input_binary_prob_multidim.target),
        (_input_binary_logit_multidim.preds, _input_binary_logit_multidim.target),
    ]
)
@pytest.mark.parametrize("normalize", ["true", "pred", "all", None])
class TestConfusionMatrix(MetricTester)
    @pytest.mark.parametrize("ddp", [True, False])
    def test_binary_confusion_matrix(self, preds, target, ddp, normalize):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryConfusionMatrix,
            sk_metric=_sk_confusion_matrix_binary,
            metric_args={
                "threshold": THRESHOLD
                "normalize": normalize
            }
        )

    def test_confusion_matrix_functional(self, preds, target, ddp, normalize):
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_confusion_matrix,
            sk_metric=_sk_confusion_matrix_binary,
            metric_args={
                "threshold": THRESHOLD
                "normalize": normalize
            }
        )

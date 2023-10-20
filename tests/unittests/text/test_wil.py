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
from typing import List, Union

import pytest
from jiwer import wil
from torchmetrics.functional.text.wil import word_information_lost
from torchmetrics.text.wil import WordInfoLost
from torchmetrics.utilities.imports import _JIWER_AVAILABLE

from unittests.text.helpers import TextTester
from unittests.text.inputs import _inputs_error_rate_batch_size_1, _inputs_error_rate_batch_size_2


def _compute_wil_metric_jiwer(preds: Union[str, List[str]], target: Union[str, List[str]]):
    return wil(target, preds)


@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        (_inputs_error_rate_batch_size_1.preds, _inputs_error_rate_batch_size_1.target),
        (_inputs_error_rate_batch_size_2.preds, _inputs_error_rate_batch_size_2.target),
    ],
)
class TestWordInfoLost(TextTester):
    """Test class for `WordInfoLost` metric."""

    @pytest.mark.parametrize("ddp", [False, True])
    def test_wil_class(self, ddp, preds, targets):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=WordInfoLost,
            reference_metric=_compute_wil_metric_jiwer,
        )

    def test_wil_functional(self, preds, targets):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=word_information_lost,
            reference_metric=_compute_wil_metric_jiwer,
        )

    def test_wil_differentiability(self, preds, targets):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=WordInfoLost,
            metric_functional=word_information_lost,
        )

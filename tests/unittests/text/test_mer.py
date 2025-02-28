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
from typing import Union

import pytest

from torchmetrics.functional.text.mer import match_error_rate
from torchmetrics.text.mer import MatchErrorRate
from unittests._helpers import seed_all
from unittests.text._helpers import TextTester
from unittests.text._inputs import _inputs_error_rate_batch_size_1, _inputs_error_rate_batch_size_2

seed_all(42)


def _reference_jiwer_mer(preds: Union[str, list[str]], target: Union[str, list[str]]):
    try:
        from jiwer import compute_measures
    except ImportError:
        pytest.skip("test requires jiwer package to be installed")
    return compute_measures(target, preds)["mer"]


@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        (_inputs_error_rate_batch_size_1.preds, _inputs_error_rate_batch_size_1.target),
        (_inputs_error_rate_batch_size_2.preds, _inputs_error_rate_batch_size_2.target),
    ],
)
class TestMatchErrorRate(TextTester):
    """Test class for `MatchErrorRate` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_mer_class(self, ddp, preds, targets):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=MatchErrorRate,
            reference_metric=_reference_jiwer_mer,
        )

    def test_mer_functional(self, preds, targets):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=match_error_rate,
            reference_metric=_reference_jiwer_mer,
        )

    def test_mer_differentiability(self, preds, targets):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=MatchErrorRate,
            metric_functional=match_error_rate,
        )

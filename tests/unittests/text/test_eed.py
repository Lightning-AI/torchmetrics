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

import pytest
from torch import Tensor, tensor
from torchmetrics.functional.text.eed import extended_edit_distance
from torchmetrics.text.eed import ExtendedEditDistance

from unittests.text.helpers import TextTester
from unittests.text.inputs import _inputs_single_reference, _inputs_single_sentence_multiple_references


def _rwth_manual_metric(preds, targets) -> Tensor:
    """Baseline implementation of metric.

    The results were obtained w.r.t. the examples defined in `tests.text.inputs` with the script from
    https://github.com/rwth-i6/ExtendedEditDistance.

    """
    ans_1 = tensor(0.24248056001808083)
    ans_2 = tensor(0.19152276295133436)

    hypothesis = "It is a guide to action which ensures that the military always obeys the commands of the party"

    # If hypothesis A and B are in preds, the average of ans_1 and ans_2 is given
    if len(preds) == 4:
        return (ans_1 + ans_2) / 2
    # If only hypothesis A or B are given, ans_1 and ans_2 are given, respectively
    if hypothesis in preds:
        return ans_1
    return ans_2


@pytest.mark.parametrize(
    ["preds", "targets"],
    [(_inputs_single_reference.preds, _inputs_single_reference.target)],
)
class TestExtendedEditDistance(TextTester):
    """Test class for `ExtendedEditDistance` metric."""

    @pytest.mark.parametrize("ddp", [False, True])
    def test_eed_class(self, preds, targets, ddp):
        """Test class implementation of metric."""
        rwth_metric = partial(_rwth_manual_metric)
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=ExtendedEditDistance,
            reference_metric=rwth_metric,
        )

    def test_eed_functional(self, preds, targets):
        """Test functional implementation of metric."""
        rwth_metric = partial(_rwth_manual_metric)
        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=extended_edit_distance,
            reference_metric=rwth_metric,
        )

    def test_eed_differentiability(self, preds, targets):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=ExtendedEditDistance,
            metric_functional=extended_edit_distance,
        )


# test blank edge cases
def test_eed_empty_functional():
    """Test that eed returns 0 when no input is provided."""
    hyp = []
    ref = [[]]
    assert extended_edit_distance(hyp, ref) == tensor(0.0)


def test_eed_empty_class():
    """Test that eed returns 0 when no input is provided."""
    eed_metric = ExtendedEditDistance()
    hyp = []
    ref = [[]]
    assert eed_metric(hyp, ref) == tensor(0.0)


def test_eed_empty_with_non_empty_hyp_functional():
    """Test that eed returns 0 when no reference is provided."""
    hyp = ["python"]
    ref = [[]]
    assert extended_edit_distance(hyp, ref) == tensor(0.0)


def test_eed_empty_with_non_empty_hyp_class():
    """Test that eed returns 0 when no reference is provided."""
    eed_metric = ExtendedEditDistance()
    hyp = ["python"]
    ref = [[]]
    assert eed_metric(hyp, ref) == tensor(0.0)


def test_eed_return_sentence_level_score_functional():
    """Test that eed can return sentence level scores."""
    hyp = _inputs_single_sentence_multiple_references.preds
    ref = _inputs_single_sentence_multiple_references.target
    _, sentence_eed = extended_edit_distance(hyp, ref, return_sentence_level_score=True)
    isinstance(sentence_eed, Tensor)


def test_eed_return_sentence_level_class():
    """Test that eed can return sentence level scores."""
    metric = ExtendedEditDistance(return_sentence_level_score=True)
    hyp = _inputs_single_sentence_multiple_references.preds
    ref = _inputs_single_sentence_multiple_references.target
    _, sentence_eed = metric(hyp, ref)
    isinstance(sentence_eed, Tensor)

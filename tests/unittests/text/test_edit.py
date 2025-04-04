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
import torch
from nltk.metrics.distance import edit_distance as nltk_edit_distance

from torchmetrics.functional.text.edit import edit_distance
from torchmetrics.text.edit import EditDistance
from unittests.text._helpers import TextTester
from unittests.text._inputs import _inputs_single_reference


@pytest.mark.parametrize(
    ("left", "right", "substitution_cost", "expected"),
    [
        ("abc", "ca", 1, 3),
        ("abc", "ca", 5, 3),
        ("wants", "wasp", 1, 3),
        ("wants", "wasp", 5, 3),
        ("rain", "shine", 1, 3),
        ("rain", "shine", 2, 5),
        ("acbdef", "abcdef", 1, 2),
        ("acbdef", "abcdef", 2, 2),
        ("lnaguaeg", "language", 1, 4),
        ("lnaguaeg", "language", 2, 4),
        ("lnaugage", "language", 1, 3),
        ("lnaugage", "language", 2, 4),
        ("lngauage", "language", 1, 2),
        ("lngauage", "language", 2, 2),
        ("wants", "swim", 1, 5),
        ("wants", "swim", 2, 7),
        ("kitten", "sitting", 1, 3),
        ("kitten", "sitting", 2, 5),
        ("duplicated", "duuplicated", 1, 1),
        ("duplicated", "duuplicated", 2, 1),
        ("very duplicated", "very duuplicateed", 2, 2),
    ],
)
def test_for_correctness(
    left: str,
    right: str,
    substitution_cost: int,
    expected,
):
    """Test the underlying implementation of edit distance.

    Test cases taken from:
    https://github.com/nltk/nltk/blob/develop/nltk/test/unit/test_distance.py

    """
    for s1, s2 in ((left, right), (right, left)):
        predicted = edit_distance(
            s1,
            s2,
            substitution_cost=substitution_cost,
        )
        assert predicted == expected


def _reference_nltk_edit_dist(preds, target, substitution_cost=1, reduction="mean"):
    costs = [nltk_edit_distance(p, t, substitution_cost=substitution_cost) for p, t in zip(preds, target)]
    if reduction == "mean":
        return sum(costs) / len(costs)
    if reduction == "sum":
        return sum(costs)
    return costs


@pytest.mark.parametrize(
    ["preds", "targets"],
    [(_inputs_single_reference.preds, _inputs_single_reference.target)],
)
class TestEditDistance(TextTester):
    """Test class for `EditDistance` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    @pytest.mark.parametrize("substitution_cost", [1, 2])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_edit_class(self, preds, targets, ddp, substitution_cost, reduction):
        """Test class implementation of metric."""
        if ddp and reduction == "none":
            pytest.skip("DDP not available for reduction='none' because order of outputs is not guaranteed.")
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=EditDistance,
            reference_metric=partial(
                _reference_nltk_edit_dist, substitution_cost=substitution_cost, reduction=reduction
            ),
            metric_args={"substitution_cost": substitution_cost, "reduction": reduction},
        )

    @pytest.mark.parametrize("substitution_cost", [1, 2])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_edit_functional(self, preds, targets, substitution_cost, reduction):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            targets=targets,
            metric_functional=edit_distance,
            reference_metric=partial(
                _reference_nltk_edit_dist, substitution_cost=substitution_cost, reduction=reduction
            ),
            metric_args={"substitution_cost": substitution_cost, "reduction": reduction},
        )

    def test_edit_differentiability(self, preds, targets):
        """Test differentiability of metric."""
        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=EditDistance,
            metric_functional=edit_distance,
        )


def test_edit_empty_functional():
    """Test functional implementation of metric with empty inputs."""
    assert edit_distance([], []) == 0


def test_edit_tensor_inputs():
    """Test edit distance with tensor inputs."""
    # Test 1D tensor inputs (single sequence)
    preds = torch.tensor([1, 2, 3, 4])
    target = torch.tensor([1, 2, 5, 4])
    assert edit_distance(preds, target) == 1.0

    # Test with different substitution cost
    assert edit_distance(preds, target, substitution_cost=2) == 2.0

    # Test 2D tensor inputs (batch of sequences)
    preds = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    target = torch.tensor([[1, 2, 5, 4], [5, 9, 7, 8]])
    # Each sequence has 1 edit, so mean is 1.0
    assert edit_distance(preds, target) == 1.0
    # Sum of edits is 2.0
    assert edit_distance(preds, target, reduction="sum") == 2.0
    assert torch.all(edit_distance(preds, target, reduction="none") == torch.tensor([1, 1], dtype=torch.int32))

    # Test with different length sequences using lists of tensors
    preds = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7])]
    target = [torch.tensor([1, 2, 3, 4]), torch.tensor([4, 5, 7])]
    # Each sequence has 1 edit, so mean is 1.0
    assert edit_distance(preds, target) == 1.0
    # Sum of edits is 2.0
    assert edit_distance(preds, target, reduction="sum") == 2.0
    assert torch.all(edit_distance(preds, target, reduction="none") == torch.tensor([1, 1], dtype=torch.int32))

    # Test with the example from the PR description
    preds = torch.tensor([[1, 2, 3], [4, 5, 6]])
    target = torch.tensor([[1, 2, 4], [4, 5, 7]])
    # Each sequence has 1 edit, so mean is 1.0
    assert edit_distance(preds, target) == 1.0

    # Test with the list example from the PR description
    gold = ["this", "is", "fun"]
    hypothesis = ["this", "isn't", "fun"]
    # When treated as lists of strings, edit distance should be 1
    assert edit_distance(gold, hypothesis) == 1.0


def test_edit_distance_class_with_tensors():
    """Test EditDistance class with tensor inputs."""
    # Test with the example from the PR description
    preds = torch.tensor([[1, 2, 3], [4, 5, 6]])
    target = torch.tensor([[1, 2, 4], [4, 5, 7]])

    # Test with default parameters
    metric = EditDistance()
    # Each sequence has 1 edit, so mean is 1.0
    assert metric(preds, target) == 1.0

    # Test with different reduction methods
    metric = EditDistance(reduction="sum")
    # Sum of edits is 2.0
    assert metric(preds, target) == 2.0

    metric = EditDistance(reduction="none")
    result = metric(preds, target)
    assert torch.all(result == torch.tensor([1, 1], dtype=torch.int32))

    # Test with different substitution cost
    metric = EditDistance(substitution_cost=2)
    # Each sequence has 1 edit with cost 2, so mean is 2.0
    assert metric(preds, target) == 2.0

    # Test accumulation over multiple batches
    metric = EditDistance()
    metric.update(preds, target)

    # Add another batch with different edit distances
    preds2 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    target2 = torch.tensor([[1, 9, 3, 4], [5, 6, 9, 8]])
    metric.update(preds2, target2)

    # Should average all edit distances (1 + 1 + 1 + 1) / 4 = 1.0
    assert metric.compute() == 1.0


def test_edit_raise_errors():
    """Test errors are raised on wrong input."""
    with pytest.raises(ValueError, match="Expected argument `substitution_cost` to be a positive integer.*"):
        EditDistance(substitution_cost=-1)

    with pytest.raises(ValueError, match="Expected argument `substitution_cost` to be a positive integer.*"):
        EditDistance(substitution_cost=2.0)

    with pytest.raises(ValueError, match="Expected argument `reduction` to be one of.*"):
        EditDistance(reduction=2.0)

    with pytest.raises(ValueError, match="Expected argument `preds` and `target` to have same length.*"):
        edit_distance(["abc"], ["abc", "def"])

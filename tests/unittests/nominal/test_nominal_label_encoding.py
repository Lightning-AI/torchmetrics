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
"""Nominal metrics must not depend on how the categories happen to be encoded.

Categorical data carries no inherent ordering or origin, so relabelling the categories consistently across ``preds``
and ``target`` must leave every nominal statistic unchanged. Before the fix for
https://github.com/Lightning-AI/torchmetrics/issues/3460 the functional metrics passed the raw labels straight to the
confusion matrix update, which requires them to be exactly ``0, ..., num_classes - 1``, so anything else raised.

"""

import pytest
import torch

from torchmetrics.functional.nominal import (
    cramers_v,
    cramers_v_matrix,
    pearsons_contingency_coefficient,
    pearsons_contingency_coefficient_matrix,
    theils_u,
    theils_u_matrix,
    tschuprows_t,
    tschuprows_t_matrix,
)
from torchmetrics.functional.nominal.utils import _compute_chi_squared

NUM_CLASSES = 4
NUM_SAMPLES = 200

METRICS = [
    pytest.param(cramers_v, id="cramers_v"),
    pytest.param(pearsons_contingency_coefficient, id="pearsons_contingency_coefficient"),
    pytest.param(theils_u, id="theils_u"),
    pytest.param(tschuprows_t, id="tschuprows_t"),
]

MATRIX_METRICS = [
    pytest.param(cramers_v_matrix, id="cramers_v_matrix"),
    pytest.param(pearsons_contingency_coefficient_matrix, id="pearsons_contingency_coefficient_matrix"),
    pytest.param(theils_u_matrix, id="theils_u_matrix"),
    pytest.param(tschuprows_t_matrix, id="tschuprows_t_matrix"),
]

# each maps the zero-based labels ``0, ..., NUM_CLASSES - 1`` onto a different encoding of the same categories
RELABELLINGS = [
    pytest.param([5, 6, 7, 8], id="offset"),
    pytest.param([1, 2, 3, 4], id="one_based"),
    pytest.param([0, 2, 7, 9], id="non_contiguous"),
    pytest.param([-3, -1, 1, 3], id="negative"),
    pytest.param([2, 0, 3, 1], id="permuted"),
]


def _zero_based_input():
    """Return a reproducible pair of zero-based categorical vectors with a non-trivial association."""
    generator = torch.Generator().manual_seed(42)
    preds = torch.randint(high=NUM_CLASSES, size=(NUM_SAMPLES,), generator=generator)
    # make target correlated with preds so the statistics are not degenerate
    noise = torch.randint(high=2, size=(NUM_SAMPLES,), generator=generator)
    target = (preds + noise) % NUM_CLASSES
    return preds, target


def _relabel(tensor: torch.Tensor, mapping: list) -> torch.Tensor:
    """Replace each zero-based label with its counterpart in ``mapping``."""
    return torch.tensor(mapping, dtype=torch.long)[tensor]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("mapping", RELABELLINGS)
def test_functional_metric_is_invariant_to_label_encoding(metric, mapping):
    """Relabelling the categories must not change the statistic."""
    preds, target = _zero_based_input()
    expected = metric(preds, target)
    actual = metric(_relabel(preds, mapping), _relabel(target, mapping))
    assert torch.allclose(actual, expected), f"{expected} != {actual} for mapping {mapping}"


@pytest.mark.parametrize("metric", MATRIX_METRICS)
@pytest.mark.parametrize("mapping", RELABELLINGS)
def test_matrix_metric_is_invariant_to_label_encoding(metric, mapping):
    """The matrix variants must be invariant to the encoding as well."""
    preds, target = _zero_based_input()
    matrix = torch.stack([preds, target], dim=-1)
    expected = metric(matrix)
    actual = metric(_relabel(matrix, mapping))
    assert torch.allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("metric", METRICS)
def test_functional_metric_accepts_float_labels(metric):
    """Float-encoded categories are supported and agree with their integer counterpart."""
    preds, target = _zero_based_input()
    expected = metric(preds, target)
    actual = metric(preds.float(), target.float())
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    "metric", [pytest.param(cramers_v, id="cramers_v"), pytest.param(tschuprows_t, id="tschuprows_t")]
)
@pytest.mark.parametrize("bias_correction", [True, False])
def test_binary_input_with_bias_correction(metric, bias_correction):
    """Binary inputs produce a 2x2 table, the only case that triggers the bias correction branch.

    The correction adds a float offset to the confusion matrix, which is integer-typed on the functional path, so
    doing it in place raised ``RuntimeError: result type Float can't be cast to the desired output type Long``.

    """
    preds = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    target = torch.tensor([0, 1, 0, 0, 1, 1, 0, 1])
    value = metric(preds, target, bias_correction=bias_correction)
    assert torch.isfinite(value) or torch.isnan(value)


def test_compute_chi_squared_does_not_mutate_input():
    """The bias correction must not write into the confusion matrix it is handed."""
    confmat = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    original = confmat.clone()
    _compute_chi_squared(confmat, bias_correction=True)
    assert torch.equal(confmat, original)


@pytest.mark.parametrize("metric", METRICS)
def test_functional_metric_raises_nothing_on_single_category(metric):
    """A single observed category is degenerate but must not raise."""
    preds = torch.full((NUM_SAMPLES,), 7)
    target = torch.full((NUM_SAMPLES,), 7)
    value = metric(preds, target)
    assert value.numel() == 1

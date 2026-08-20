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


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (cramers_v, {"bias_correction": False}),
        (pearsons_contingency_coefficient, {}),
        (theils_u, {}),
        (tschuprows_t, {"bias_correction": False}),
    ],
)
def test_functional_metric_is_invariant_to_category_labels(metric, kwargs):
    """Test that functional nominal metrics support non-contiguous category labels."""
    categories = torch.tensor([0, 0, 1, 1])
    relabeled_categories = torch.tensor([2, 2, 5, 5])

    expected = metric(categories, categories, **kwargs)
    actual = metric(relabeled_categories, relabeled_categories, **kwargs)

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    ("metric", "kwargs"),
    [
        (cramers_v_matrix, {"bias_correction": False}),
        (pearsons_contingency_coefficient_matrix, {}),
        (theils_u_matrix, {}),
        (tschuprows_t_matrix, {"bias_correction": False}),
    ],
)
def test_matrix_metric_is_invariant_to_category_labels(metric, kwargs):
    """Test that nominal matrix metrics support different non-contiguous labels per variable."""
    categories = torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]])
    relabeled_categories = categories * torch.tensor([3, 4]) + torch.tensor([2, 7])

    expected = metric(categories, **kwargs)
    actual = metric(relabeled_categories, **kwargs)

    assert torch.allclose(actual, expected)

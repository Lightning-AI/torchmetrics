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

import numpy as np
import pytest
import torch
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from torchmetrics.functional.regression.cosine_similarity import cosine_similarity
from torchmetrics.regression.cosine_similarity import CosineSimilarity
from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)

NUM_TARGETS = 5


_single_target_inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, 1),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, 1),
)

_multi_target_inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_TARGETS),
)


def _reference_sklearn_cosine(preds, target, reduction):
    sk_preds = preds.numpy()
    sk_target = target.numpy()
    result_array = sk_cosine(sk_target, sk_preds)
    col = np.diagonal(result_array)
    col_sum = col.sum()
    if reduction == "sum":
        return col_sum
    if reduction == "mean":
        return col_sum / len(col)
    return col


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize(
    ("preds", "target", "ref_metric"),
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _reference_sklearn_cosine),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _reference_sklearn_cosine),
    ],
)
class TestCosineSimilarity(MetricTester):
    """Test class for `CosineSimilarity` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_cosine_similarity(self, reduction, preds, target, ref_metric, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            CosineSimilarity,
            partial(ref_metric, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    def test_cosine_similarity_functional(self, reduction, preds, target, ref_metric):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            cosine_similarity,
            partial(ref_metric, reduction=reduction),
            metric_args={"reduction": reduction},
        )


def test_error_on_different_shape(metric_class=CosineSimilarity):
    """Test that error is raised on different shapes of input."""
    metric = metric_class()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100, 2), torch.randn(50, 2))


def test_error_on_non_2d_input():
    """Test that error is raised if input is not 2-dimensional."""
    metric = CosineSimilarity()
    with pytest.raises(ValueError, match="Expected input to cosine similarity to be 2D tensors of shape.*"):
        metric(torch.randn(100), torch.randn(100))


@pytest.mark.parametrize("scale", [1.0, 1e20, 1e-30, 1e100])
def test_cosine_similarity_is_scale_invariant(scale):
    """A finite nonzero vector is cosine-similar to itself at any scale.

    The squared norms used to overflow at large scales and underflow at small ones, so the
    ratio came back as inf/inf or 0/0 and both APIs returned NaN for finite inputs.
    """
    x = torch.tensor([[scale, scale]], dtype=torch.float64)

    functional = cosine_similarity(x, x, "none")
    metric = CosineSimilarity(reduction="none")(x, x)

    assert torch.isfinite(functional).all()
    assert torch.isfinite(metric).all()
    assert torch.allclose(functional, torch.ones_like(functional))
    assert torch.allclose(metric, torch.ones_like(metric))


@pytest.mark.parametrize("scale", [1.0, 1e20, 1e-30])
@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ([1.0, 2.0], [2.0, 1.0], 0.8),
        ([-2.0, 1.0], [2.0, 1.0], -0.6),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
    ],
)
def test_cosine_similarity_closed_form_at_extreme_scales(first, second, expected, scale):
    """Textbook values stay exact when both vectors are scaled to an extreme magnitude."""
    preds = torch.tensor([first], dtype=torch.float64) * scale
    target = torch.tensor([second], dtype=torch.float64) * scale

    assert torch.allclose(cosine_similarity(preds, target, "none"), torch.tensor([expected], dtype=torch.float64))


def test_cosine_similarity_preserves_float64():
    """The update used to cast both inputs to float32 unconditionally."""
    preds = torch.tensor([[1.0, 2.0]], dtype=torch.float64)

    assert cosine_similarity(preds, preds, "none").dtype == torch.float64
    assert CosineSimilarity(reduction="none")(preds, preds).dtype == torch.float64


def test_cosine_similarity_keeps_half_precision_in_float32():
    """Half precision has no room for the norm intermediates, so it stays promoted."""
    preds = torch.tensor([[1.0, 2.0]], dtype=torch.float16)

    assert cosine_similarity(preds, preds, "none").dtype == torch.float32

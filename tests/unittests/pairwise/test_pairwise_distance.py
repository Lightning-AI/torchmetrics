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
from collections import namedtuple
from functools import partial

import pytest
import torch
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    linear_kernel,
    manhattan_distances,
    pairwise_distances,
)
from torchmetrics.functional import (
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
    pairwise_linear_similarity,
    pairwise_manhattan_distance,
    pairwise_minkowski_distance,
)

from unittests import BATCH_SIZE, NUM_BATCHES
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

extra_dim = 5

Input = namedtuple("Input", ["x", "y"])


_inputs1 = Input(
    x=torch.rand(NUM_BATCHES, BATCH_SIZE, extra_dim),
    y=torch.rand(NUM_BATCHES, BATCH_SIZE, extra_dim),
)


_inputs2 = Input(
    x=torch.rand(NUM_BATCHES, BATCH_SIZE, extra_dim),
    y=torch.rand(NUM_BATCHES, BATCH_SIZE, extra_dim),
)


def _wrap_reduction(x, y, sk_fn, reduction):
    x = x.view(-1, extra_dim).numpy()
    y = y.view(-1, extra_dim).numpy()
    res = sk_fn(x, y)
    if reduction == "sum":
        return res.sum(axis=-1)
    if reduction == "mean":
        return res.mean(axis=-1)
    return res


@pytest.mark.parametrize(
    "x, y",
    [
        (_inputs1.x, _inputs1.y),
        (_inputs2.x, _inputs2.y),
    ],
)
@pytest.mark.parametrize(
    "metric_functional, sk_fn",
    [
        pytest.param(pairwise_cosine_similarity, cosine_similarity, id="cosine"),
        pytest.param(pairwise_euclidean_distance, euclidean_distances, id="euclidean"),
        pytest.param(pairwise_manhattan_distance, manhattan_distances, id="manhatten"),
        pytest.param(pairwise_linear_similarity, linear_kernel, id="linear"),
        pytest.param(
            partial(pairwise_minkowski_distance, exponent=3),
            partial(pairwise_distances, metric="minkowski", p=3),
            id="minkowski-3",
        ),
        pytest.param(
            partial(pairwise_minkowski_distance, exponent=4),
            partial(pairwise_distances, metric="minkowski", p=4),
            id="minkowski-4",
        ),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", None])
class TestPairwise(MetricTester):
    """Test pairwise implementations."""

    atol = 1e-4

    def test_pairwise_functional(self, x, y, metric_functional, sk_fn, reduction):
        """Test functional pairwise implementations."""
        self.run_functional_metric_test(
            preds=x,
            target=y,
            metric_functional=metric_functional,
            reference_metric=partial(_wrap_reduction, sk_fn=sk_fn, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    def test_pairwise_half_cpu(self, x, y, metric_functional, sk_fn, reduction, request):
        """Test half precision support on cpu."""
        if "euclidean" in request.node.callspec.id:
            pytest.xfail("pairwise_euclidean_distance metric does not support cpu + half precision")
        self.run_precision_test_cpu(x, y, None, metric_functional, metric_args={"reduction": reduction})

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_pairwise_half_gpu(self, x, y, metric_functional, sk_fn, reduction):
        """Test half precision support on gpu."""
        self.run_precision_test_gpu(x, y, None, metric_functional, metric_args={"reduction": reduction})


@pytest.mark.parametrize(
    "metric",
    [
        pairwise_cosine_similarity,
        pairwise_euclidean_distance,
        pairwise_manhattan_distance,
        partial(pairwise_minkowski_distance, exponent=3),
    ],
)
def test_error_on_wrong_shapes(metric):
    """Test errors are raised on wrong input."""
    with pytest.raises(ValueError, match="Expected argument `x` to be a 2D tensor .*"):
        metric(torch.randn(10))

    with pytest.raises(ValueError, match="Expected argument `y` to be a 2D tensor .*"):
        metric(torch.randn(10, 5), torch.randn(5, 3))

    with pytest.raises(ValueError, match="Expected reduction to be one of .*"):
        metric(torch.randn(10, 5), torch.randn(10, 5), reduction=1)


@pytest.mark.parametrize(
    ("metric_functional", "sk_fn"),
    [
        (pairwise_cosine_similarity, cosine_similarity),
        (pairwise_euclidean_distance, euclidean_distances),
        (pairwise_manhattan_distance, manhattan_distances),
        (pairwise_linear_similarity, linear_kernel),
        (partial(pairwise_minkowski_distance, exponent=3), partial(pairwise_distances, metric="minkowski", p=3)),
    ],
)
def test_precision_case(metric_functional, sk_fn):
    """Test that metrics are robust towars cases where high precision is needed."""
    x = torch.tensor([[772.0, 112.0], [772.20001, 112.0]])
    res1 = metric_functional(x, zero_diagonal=False)
    res2 = sk_fn(x)
    assert torch.allclose(res1, torch.tensor(res2, dtype=torch.float32))

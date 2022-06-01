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
from collections import namedtuple
from functools import partial

import pytest
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, linear_kernel, manhattan_distances

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.functional import (
    pairwise_cosine_similarity,
    pairwise_euclidean_distance,
    pairwise_linear_similarity,
    pairwise_manhattan_distance,
)
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_7

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


def _sk_metric(x, y, sk_fn, reduction):
    """comparison function."""
    x = x.view(-1, extra_dim).numpy()
    y = y.view(-1, extra_dim).numpy()
    res = sk_fn(x, y)
    if reduction == "sum":
        return res.sum(axis=-1)
    elif reduction == "mean":
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
        (pairwise_cosine_similarity, cosine_similarity),
        (pairwise_euclidean_distance, euclidean_distances),
        (pairwise_manhattan_distance, manhattan_distances),
        (pairwise_linear_similarity, linear_kernel),
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", None])
class TestPairwise(MetricTester):
    """test pairwise implementations."""

    atol = 1e-4

    def test_pairwise_functional(self, x, y, metric_functional, sk_fn, reduction):
        """test functional pairwise implementations."""
        self.run_functional_metric_test(
            preds=x,
            target=y,
            metric_functional=metric_functional,
            sk_metric=partial(_sk_metric, sk_fn=sk_fn, reduction=reduction),
            metric_args={"reduction": reduction},
        )

    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_1_7, reason="half support of core operations on not support before pytorch v1.7"
    )
    def test_pairwise_half_cpu(self, x, y, metric_functional, sk_fn, reduction):
        """test half precision support on cpu."""
        if metric_functional == pairwise_euclidean_distance:
            pytest.xfail("pairwise_euclidean_distance metric does not support cpu + half precision")
        self.run_precision_test_cpu(x, y, None, metric_functional, metric_args={"reduction": reduction})

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_pairwise_half_gpu(self, x, y, metric_functional, sk_fn, reduction):
        """test half precision support on gpu."""
        self.run_precision_test_gpu(x, y, None, metric_functional, metric_args={"reduction": reduction})


@pytest.mark.parametrize(
    "metric", [pairwise_cosine_similarity, pairwise_euclidean_distance, pairwise_manhattan_distance]
)
def test_error_on_wrong_shapes(metric):
    """Test errors are raised on wrong input."""
    with pytest.raises(ValueError, match="Expected argument `x` to be a 2D tensor .*"):
        metric(torch.randn(10))

    with pytest.raises(ValueError, match="Expected argument `y` to be a 2D tensor .*"):
        metric(torch.randn(10, 5), torch.randn(5, 3))

    with pytest.raises(ValueError, match="Expected reduction to be one of .*"):
        metric(torch.randn(10, 5), torch.randn(10, 5), reduction=1)

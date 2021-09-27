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
import math
from collections import namedtuple
from functools import partial

import pytest
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.functional import (
    pairwise_euclidean_distance,
    pairwise_cosine_similarity,
    pairwise_manhatten_distance
)
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6

seed_all(42)

extra_dim = 5

Input = namedtuple('Input', ["X", "Y"])


_inputs1 = Input(
    X=torch.rand(NUM_BATCHES, BATCH_SIZE, extra_dim),
    Y=torch.rand(NUM_BATCHES, BATCH_SIZE, extra_dim),
)


_inputs2 = Input(
    X=torch.rand(NUM_BATCHES, BATCH_SIZE, extra_dim),
    Y=torch.rand(NUM_BATCHES, BATCH_SIZE, extra_dim),
)


def _sk_metric(X, Y, sk_fn, reduction):
    X = X.view(-1, extra_dim).numpy()
    Y = Y.view(-1, extra_dim).numpy()
    res = sk_fn(X, Y)
    if reduction == 'sum':
        return res.sum(axis=-1)
    elif reduction == 'mean':
        return res.mean(axis=-1)
    return res


@pytest.mark.parametrize("X, Y",
    [
        (_inputs1.X, _inputs1.Y),
        (_inputs1.X, _inputs1.Y),
    ],
)
@pytest.mark.parametrize("metric_functional, sk_fn",
    [
        (pairwise_cosine_similarity, cosine_similarity),
        (pairwise_euclidean_distance, euclidean_distances),
        (pairwise_manhatten_distance, manhattan_distances)
    ],
)
@pytest.mark.parametrize("reduction", ["sum", "mean", None])
class TestPairwise(MetricTester):
    def test_pairwise_functional(self, X, Y, metric_functional, sk_fn, reduction):
        # todo: `metric_class` is unused
        self.run_functional_metric_test(
            preds=X,
            target=Y,
            metric_functional=metric_functional,
            sk_metric=partial(_sk_metric, sk_fn=sk_fn, reduction=reduction),
            metric_args={'reduction': reduction}
        )


@pytest.mark.parametrize(
    "metric", [pairwise_cosine_similarity, pairwise_euclidean_distance, pairwise_manhatten_distance]
)
def test_error_on_wrong_shapes(metric):
    with pytest.raises(ValueError, match='Expected argument `X` to be a 2D tensor .*'):
        metric(torch.randn(10))

    with pytest.raises(ValueError, match='Expected argument `Y` to be a 2D tensor .*'):
        metric(torch.randn(10, 5), torch.randn(5, 3))

    with pytest.raises(ValueError, match='Expected reduction to be one of -*'):
        metric(torch.randn(10, 5), torch.randn(10, 5), reduction=1)

    
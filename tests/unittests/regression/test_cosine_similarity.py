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
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

num_targets = 5


_single_target_inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE),
)

_multi_target_inputs = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
    target=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
)


def _multi_target_ref_metric(preds, target, reduction, sk_fn=sk_cosine):
    sk_preds = preds.view(-1, num_targets).numpy()
    sk_target = target.view(-1, num_targets).numpy()
    result_array = sk_fn(sk_target, sk_preds)
    col = np.diagonal(result_array)
    col_sum = col.sum()
    if reduction == "sum":
        return col_sum
    if reduction == "mean":
        return col_sum / len(col)
    return col


def _single_target_ref_metric(preds, target, reduction, sk_fn=sk_cosine):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    result_array = sk_fn(np.expand_dims(sk_preds, axis=0), np.expand_dims(sk_target, axis=0))
    col = np.diagonal(result_array)
    col_sum = col.sum()
    if reduction == "sum":
        return col_sum
    if reduction == "mean":
        return col_sum / len(col)
    return col


@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize(
    "preds, target, ref_metric",
    [
        (_single_target_inputs.preds, _single_target_inputs.target, _single_target_ref_metric),
        (_multi_target_inputs.preds, _multi_target_inputs.target, _multi_target_ref_metric),
    ],
)
class TestCosineSimilarity(MetricTester):
    """Test class for `CosineSimilarity` metric."""

    @pytest.mark.parametrize("ddp", [True, False])
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
        metric(torch.randn(100), torch.randn(50))

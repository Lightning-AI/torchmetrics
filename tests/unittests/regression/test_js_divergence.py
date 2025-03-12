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
from typing import NamedTuple, Optional

import numpy as np
import pytest
import torch
from scipy.spatial.distance import jensenshannon
from torch import Tensor

from torchmetrics.functional.regression.js_divergence import jensen_shannon_divergence
from torchmetrics.regression.js_divergence import JensenShannonDivergence
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)


class _Input(NamedTuple):
    p: Tensor
    q: Tensor


_probs_inputs = _Input(
    p=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
    q=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
)

_log_probs_inputs = _Input(
    p=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM).softmax(dim=-1).log(),
    q=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM).softmax(dim=-1).log(),
)


def _wrap_reduction(p: Tensor, q: Tensor, log_prob: bool, reduction: Optional[str] = "mean"):
    if log_prob:
        p = p.softmax(dim=-1)
        q = q.softmax(dim=-1)
    res = jensenshannon(p, q, axis=1) ** 2
    if reduction == "mean":
        return np.mean(res)
    if reduction == "sum":
        return np.sum(res)
    return res


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize(
    "p, q, log_prob", [(_probs_inputs.p, _probs_inputs.q, False), (_log_probs_inputs.p, _log_probs_inputs.q, True)]
)
class TestJensenShannonDivergence(MetricTester):
    """Test class for `JensenShannonDivergence` metric."""

    atol = 1e-6

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_jensen_shannon_divergence(self, reduction, p, q, log_prob, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            p,
            q,
            JensenShannonDivergence,
            partial(_wrap_reduction, log_prob=log_prob, reduction=reduction),
            metric_args={"log_prob": log_prob, "reduction": reduction},
        )

    def test_jensen_shannon_divergence_functional(self, reduction, p, q, log_prob):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            p,
            q,
            jensen_shannon_divergence,
            partial(_wrap_reduction, log_prob=log_prob, reduction=reduction),
            metric_args={"log_prob": log_prob, "reduction": reduction},
        )

    def test_jensen_shannon_divergence_differentiability(self, reduction, p, q, log_prob):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            p,
            q,
            metric_module=JensenShannonDivergence,
            metric_functional=jensen_shannon_divergence,
            metric_args={"log_prob": log_prob, "reduction": reduction},
        )

    # JensenShannonDivergence half + cpu does not work due to missing support in torch.clamp
    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_2_1,
        reason="Pytoch below 2.1 does not support cpu + half precision used in JensenShannonDivergence metric",
    )
    def test_jensen_shannon_divergence_half_cpu(self, reduction, p, q, log_prob):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            p, q, JensenShannonDivergence, jensen_shannon_divergence, {"log_prob": log_prob, "reduction": reduction}
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_jensen_shannon_divergence_half_gpu(self, reduction, p, q, log_prob):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            p, q, JensenShannonDivergence, jensen_shannon_divergence, {"log_prob": log_prob, "reduction": reduction}
        )


def test_error_on_different_shape():
    """Test that error is raised on different shapes of input."""
    metric = JensenShannonDivergence()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))


def test_error_on_multidim_tensors():
    """Test that error is raised if a larger than 2D tensor is given as input."""
    metric = JensenShannonDivergence()
    with pytest.raises(ValueError, match="Expected both p and q distribution to be 2D but got 3 and 3 respectively"):
        metric(torch.randn(10, 20, 5), torch.randn(10, 20, 5))


def test_zero_probability():
    """When p = 0 in Jensen-Shannon divergence the score should not output Nan."""
    metric = JensenShannonDivergence()
    metric.update(
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        torch.tensor(torch.randn(3, 3).softmax(dim=-1)),
    )
    assert not torch.isnan(metric.compute())

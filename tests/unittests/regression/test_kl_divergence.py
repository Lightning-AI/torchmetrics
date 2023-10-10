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
from typing import Optional

import numpy as np
import pytest
import torch
from scipy.stats import entropy
from torch import Tensor
from torchmetrics.functional.regression.kl_divergence import kl_divergence
from torchmetrics.regression.kl_divergence import KLDivergence
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1

from unittests import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

Input = namedtuple("Input", ["p", "q"])

_probs_inputs = Input(
    p=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
    q=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM),
)

_log_probs_inputs = Input(
    p=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM).softmax(dim=-1).log(),
    q=torch.rand(NUM_BATCHES, BATCH_SIZE, EXTRA_DIM).softmax(dim=-1).log(),
)


def _wrap_reduction(p: Tensor, q: Tensor, log_prob: bool, reduction: Optional[str] = "mean"):
    if log_prob:
        p = p.softmax(dim=-1)
        q = q.softmax(dim=-1)
    res = entropy(p, q, axis=1)
    if reduction == "mean":
        return np.mean(res)
    if reduction == "sum":
        return np.sum(res)
    return res


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize(
    "p, q, log_prob", [(_probs_inputs.p, _probs_inputs.q, False), (_log_probs_inputs.p, _log_probs_inputs.q, True)]
)
class TestKLDivergence(MetricTester):
    """Test class for `KLDivergence` metric."""

    atol = 1e-6

    @pytest.mark.parametrize("ddp", [True, False])
    def test_kldivergence(self, reduction, p, q, log_prob, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            p,
            q,
            KLDivergence,
            partial(_wrap_reduction, log_prob=log_prob, reduction=reduction),
            metric_args={"log_prob": log_prob, "reduction": reduction},
        )

    def test_kldivergence_functional(self, reduction, p, q, log_prob):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            p,
            q,
            kl_divergence,
            partial(_wrap_reduction, log_prob=log_prob, reduction=reduction),
            metric_args={"log_prob": log_prob, "reduction": reduction},
        )

    def test_kldivergence_differentiability(self, reduction, p, q, log_prob):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            p,
            q,
            metric_module=KLDivergence,
            metric_functional=kl_divergence,
            metric_args={"log_prob": log_prob, "reduction": reduction},
        )

    # KLDivergence half + cpu does not work due to missing support in torch.clamp
    @pytest.mark.skipif(
        not _TORCH_GREATER_EQUAL_2_1,
        reason="Pytoch below 2.1 does not support cpu + half precision used in KLDivergence metric",
    )
    def test_kldivergence_half_cpu(self, reduction, p, q, log_prob):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(p, q, KLDivergence, kl_divergence, {"log_prob": log_prob, "reduction": reduction})

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_kldivergence_half_gpu(self, reduction, p, q, log_prob):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(p, q, KLDivergence, kl_divergence, {"log_prob": log_prob, "reduction": reduction})


def test_error_on_different_shape():
    """Test that error is raised on different shapes of input."""
    metric = KLDivergence()
    with pytest.raises(RuntimeError, match="Predictions and targets are expected to have the same shape"):
        metric(torch.randn(100), torch.randn(50))


def test_error_on_multidim_tensors():
    """Test that error is raised if a larger than 2D tensor is given as input."""
    metric = KLDivergence()
    with pytest.raises(ValueError, match="Expected both p and q distribution to be 2D but got 3 and 3 respectively"):
        metric(torch.randn(10, 20, 5), torch.randn(10, 20, 5))


def test_zero_probability():
    """When p = 0 in kl divergence the score should not output Nan."""
    metric = KLDivergence()
    metric.update(
        torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        torch.tensor(torch.randn(3, 3).softmax(dim=-1)),
    )
    assert not torch.isnan(metric.compute())


def test_inf_case():
    """When q = 0 in kl divergence the score should be inf."""
    metric = KLDivergence()
    metric.update(torch.tensor([[0.3, 0.3, 0.4]]), torch.tensor([[0.5, 0.5, 0]]))
    assert not torch.isfinite(metric.compute())

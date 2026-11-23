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
from torch.nn import functional as F  # noqa: N812

from torchmetrics.functional.text.perplexity import perplexity
from torchmetrics.text.perplexity import Perplexity
from unittests._helpers.testers import MetricTester
from unittests.text._inputs import (
    MASK_INDEX,
    _logits_inputs_fp32,
    _logits_inputs_fp32_with_mask,
    _logits_inputs_fp64,
    _logits_inputs_fp64_with_mask,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("ignore_index", [None, -100])
def test_perplexity_extreme_logits(dtype, ignore_index):
    """A rare target token must not turn a finite mean cross entropy into infinite perplexity."""
    magnitude = 20.0 if dtype == torch.float16 else 500.0
    preds = torch.empty(2, 10, 2, dtype=dtype)
    preds[..., 0], preds[..., 1] = magnitude, -magnitude
    target = torch.zeros(2, 10, dtype=torch.long)
    target[-1, -1] = 1
    if ignore_index is not None:
        target[0, :2] = ignore_index
    preds.requires_grad_()
    expected = F.cross_entropy(preds.reshape(-1, 2), target.reshape(-1), ignore_index=-100).exp()
    assert torch.isfinite(expected)
    result = perplexity(preds, target, ignore_index)
    torch.testing.assert_close(result, expected)
    assert torch.isfinite(torch.autograd.grad(result, preds)[0]).all()
    metric = Perplexity(ignore_index=ignore_index).to(dtype)
    for pred_batch, target_batch in zip(preds.split(1), target.split(1)):
        metric.update(pred_batch, target_batch)
    torch.testing.assert_close(metric.compute(), expected.detach())


@pytest.mark.parametrize("invalid_index", [-1, 2])
def test_perplexity_invalid_class_indices(invalid_index):
    """Class indices outside the vocabulary must raise instead of indexing from the end."""
    with pytest.raises((IndexError, RuntimeError), match="out of bounds"):
        perplexity(torch.zeros(1, 1, 2), torch.tensor([[invalid_index]]))


def test_perplexity_ignored_nonfinite_logits():
    """Ignored logits must contribute neither loss nor undefined gradients."""
    preds = torch.tensor([[[float("nan"), float("nan")], [0.0, 0.0]]], requires_grad=True)
    target = torch.tensor([[-100, 0]])
    result = perplexity(preds, target, ignore_index=-100)
    torch.testing.assert_close(result, torch.tensor(2.0))
    assert torch.isfinite(torch.autograd.grad(result, preds)[0]).all()


def _reference_local_perplexity(preds, target, ignore_index):
    """Baseline implementation of perplexity metric based upon PyTorch Cross Entropy."""
    preds = preds.reshape(-1, preds.shape[-1])
    target = target.reshape(-1)
    cross_entropy = F.cross_entropy(preds, target)
    return torch.exp(cross_entropy)


@pytest.mark.parametrize(
    ("preds", "target", "ignore_index"),
    [
        (_logits_inputs_fp32.preds, _logits_inputs_fp32.target, None),
        (_logits_inputs_fp64.preds, _logits_inputs_fp64.target, None),
        (_logits_inputs_fp32_with_mask.preds, _logits_inputs_fp32_with_mask.target, MASK_INDEX),
        (_logits_inputs_fp64_with_mask.preds, _logits_inputs_fp64_with_mask.target, MASK_INDEX),
    ],
)
class TestPerplexity(MetricTester):
    """Test class for `Perplexity` metric."""

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_perplexity_class(self, ddp, preds, target, ignore_index):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=Perplexity,
            reference_metric=partial(_reference_local_perplexity, ignore_index=ignore_index),
            metric_args={"ignore_index": ignore_index},
        )

    def test_perplexity_fn(self, preds, target, ignore_index):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=perplexity,
            reference_metric=partial(_reference_local_perplexity, ignore_index=ignore_index),
            metric_args={"ignore_index": ignore_index},
        )

    def test_perplexity_differentiability(self, preds, target, ignore_index):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=Perplexity,
            metric_functional=perplexity,
            metric_args={"ignore_index": ignore_index},
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_perplexity_dtypes_cpu(self, preds, target, ignore_index, dtype):
        """Test dtype support of the metric on CPU."""
        self.run_precision_test_cpu(
            preds, target, Perplexity, perplexity, metric_args={"ignore_index": ignore_index}, dtype=dtype
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_perplexity_dtypes_gpu(self, preds, target, ignore_index, dtype):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds, target, Perplexity, perplexity, metric_args={"ignore_index": ignore_index}, dtype=dtype
        )

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
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_2
from unittests._helpers.testers import MetricTester
from unittests.text._inputs import (
    MASK_INDEX,
    _logits_inputs_fp32,
    _logits_inputs_fp32_with_mask,
    _logits_inputs_fp64,
    _logits_inputs_fp64_with_mask,
)


def _reference_local_perplexity(preds, target, ignore_index):
    """Baseline implementation of perplexity metric based upon PyTorch Cross Entropy."""
    preds = preds.reshape(-1, preds.shape[-1])
    target = target.reshape(-1)
    cross_entropy = F.cross_entropy(preds, target)
    return torch.exp(cross_entropy)


@pytest.mark.parametrize(
    "preds, target, ignore_index",
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
        if dtype == torch.half and not _TORCH_GREATER_EQUAL_2_2:
            with pytest.raises(RuntimeError, match="\"softmax_lastdim_kernel_impl\" not implemented for 'Half'"):
                self.run_precision_test_cpu(
                    preds, target, Perplexity, perplexity, metric_args={"ignore_index": ignore_index}, dtype=dtype
                )
        else:
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

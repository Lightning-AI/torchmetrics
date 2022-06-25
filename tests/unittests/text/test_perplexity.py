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
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from torchmetrics.functional.text.perplexity import perplexity
from torchmetrics.text.perplexity import Perplexity
from unittests.helpers.testers import MetricTester
from unittests.text.inputs import (
    MASK_INDEX,
    _logits_inputs_fp32,
    _logits_inputs_fp32_with_mask,
    _logits_inputs_fp64,
    _logits_inputs_fp64_with_mask,
)


def _sk_perplexity(preds, target, ignore_index):
    """Reference Perplexity metrics based upon PyTorch Cross Entropy."""
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
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_perplexity_class(self, ddp, dist_sync_on_step, preds, target, ignore_index):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=Perplexity,
            sk_metric=partial(_sk_perplexity, ignore_index=ignore_index),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"ignore_index": ignore_index},
        )

    def test_perplexity_fn(self, preds, target, ignore_index):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=perplexity,
            sk_metric=partial(_sk_perplexity, ignore_index=ignore_index),
            metric_args={"ignore_index": ignore_index},
        )

    def test_accuracy_differentiability(self, preds, target, ignore_index):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=Perplexity,
            metric_functional=perplexity,
            metric_args={"ignore_index": ignore_index},
        )

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
from pytorch_msssim import ms_ssim
from torchmetrics.functional.image.ssim import multiscale_structural_similarity_index_measure
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure

from unittests import NUM_BATCHES, _Input
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


BATCH_SIZE = 1

_inputs = []
for size, coef in [(182, 0.9), (182, 0.7)]:
    preds = torch.rand(NUM_BATCHES, BATCH_SIZE, 1, size, size)
    _inputs.append(
        _Input(
            preds=preds,
            target=preds * coef,
        )
    )


def _pytorch_ms_ssim(preds, target, data_range, kernel_size):
    return ms_ssim(preds, target, data_range=data_range, win_size=kernel_size, size_average=False)


@pytest.mark.parametrize(
    "preds, target",
    [(i.preds, i.target) for i in _inputs],
)
class TestMultiScaleStructuralSimilarityIndexMeasure(MetricTester):
    """Test class for `MultiScaleStructuralSimilarityIndexMeasure` metric."""

    atol = 6e-3

    # in the pytorch-msssim package, sigma is hardcoded to 1.5. We can thus only test this value, which corresponds
    # to a kernel size of 11

    @pytest.mark.parametrize("ddp", [False, True])
    def test_ms_ssim(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            MultiScaleStructuralSimilarityIndexMeasure,
            partial(_pytorch_ms_ssim, data_range=1.0, kernel_size=11),
            metric_args={"data_range": 1.0, "kernel_size": 11},
        )

    def test_ms_ssim_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            multiscale_structural_similarity_index_measure,
            partial(_pytorch_ms_ssim, data_range=1.0, kernel_size=11),
            metric_args={"data_range": 1.0, "kernel_size": 11},
        )

    def test_ms_ssim_differentiability(self, preds, target):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        # We need to minimize this example to make the test tractable
        single_beta = (1.0,)
        _preds = preds[:, :, :, :16, :16]
        _target = target[:, :, :, :16, :16]

        self.run_differentiability_test(
            _preds.type(torch.float64),
            _target.type(torch.float64),
            metric_functional=multiscale_structural_similarity_index_measure,
            metric_module=MultiScaleStructuralSimilarityIndexMeasure,
            metric_args={
                "data_range": 1.0,
                "kernel_size": 11,
                "betas": single_beta,
            },
        )


def test_ms_ssim_contrast_sensitivity():
    """Test that the contrast sensitivity is correctly computed with 3d input."""
    preds = torch.rand(1, 1, 50, 50, 50)
    target = torch.rand(1, 1, 50, 50, 50)
    out = multiscale_structural_similarity_index_measure(
        preds, target, data_range=1.0, kernel_size=3, betas=(1.0, 0.5, 0.25)
    )
    assert isinstance(out, torch.Tensor)

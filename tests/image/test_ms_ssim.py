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
from pytorch_msssim import msssim

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics.functional.image.ms_ssim import ms_ssim
from torchmetrics.image.ms_ssim import MultiScaleSSIM

seed_all(42)

Input = namedtuple("Input", ["preds", "target"])

_inputs = []
for size, coef in [(128, 0.9), (128, 0.7)]:
    preds = torch.rand(NUM_BATCHES, BATCH_SIZE, 1, size, size)
    _inputs.append(
        Input(
            preds=preds,
            target=preds * coef,
        )
    )


def pytorch_msssim(preds, target, val_range, kernel_size, normalize):
    return msssim(img1=preds, img2=target, val_range=val_range, window_size=kernel_size, normalize=normalize)


@pytest.mark.parametrize(
    "preds, target",
    [(i.preds, i.target) for i in _inputs],
)
@pytest.mark.parametrize(
    ["kernel_size", "normalize"],
    [
        (3, None),
        (5, "relu"),
        (7, "simple"),
    ],
)
class TestMultiScaleSSIM(MetricTester):
    atol = 6e-3

    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_ms_ssim(self, preds, target, kernel_size, normalize, ddp, dist_sync_on_step):
        print(preds.size)
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            MultiScaleSSIM,
            partial(pytorch_msssim, val_range=1.0, kernel_size=kernel_size, normalize=normalize),
            metric_args={"data_range": 1.0, "kernel_size": (kernel_size, kernel_size), "normalize": normalize},
            dist_sync_on_step=dist_sync_on_step,
        )

    def test_ms_ssim_functional(self, preds, target, kernel_size, normalize):
        self.run_functional_metric_test(
            preds,
            target,
            ms_ssim,
            partial(pytorch_msssim, val_range=1.0, kernel_size=kernel_size, normalize=normalize),
            metric_args={"data_range": 1.0, "kernel_size": (kernel_size, kernel_size), "normalize": normalize},
        )

    def test_ms_ssim_differentiability(self, preds, target, kernel_size, normalize):
        # We need to minimize this example to make the test tractable
        single_beta = (1.0,)
        _preds = preds[:, :, :, :16, :16]
        _target = target[:, :, :, :16, :16]

        self.run_differentiability_test(
            _preds.type(torch.float64),
            _target.type(torch.float64),
            metric_functional=ms_ssim,
            metric_module=MultiScaleSSIM,
            metric_args={
                "data_range": 1.0,
                "kernel_size": (kernel_size, kernel_size),
                "normalize": normalize,
                "betas": single_beta,
            },
        )

    # SSIM half + cpu does not work due to missing support in torch.log
    @pytest.mark.xfail(reason="SSIM metric does not support cpu + half precision")
    def test_ms_ssim_half_cpu(self, preds, target, kernel_size, normalize):
        self.run_precision_test_cpu(
            preds,
            target,
            MultiScaleSSIM,
            ms_ssim,
            metric_args={"data_range": 1.0, "kernel_size": (kernel_size, kernel_size), "normalize": normalize},
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_ms_ssim_half_gpu(self, preds, target, kernel_size, normalize):
        self.run_precision_test_gpu(
            preds,
            target,
            MultiScaleSSIM,
            ms_ssim,
            metric_args={"data_range": 1.0, "kernel_size": (kernel_size, kernel_size), "normalize": normalize},
        )

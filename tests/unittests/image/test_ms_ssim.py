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

import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pytorch_msssim import ms_ssim

from torchmetrics.functional.image.ssim import multiscale_structural_similarity_index_measure
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure
from unittests import NUM_BATCHES, _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.conftest import MAX_PORT, START_PORT

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


def _reference_ms_ssim(preds, target, data_range: float = 1.0, kernel_size: int = 11):
    return ms_ssim(preds, target, data_range=data_range, win_size=kernel_size, size_average=False)


@pytest.mark.parametrize(
    ("preds", "target"),
    [(i.preds, i.target) for i in _inputs],
)
class TestMultiScaleStructuralSimilarityIndexMeasure(MetricTester):
    """Test class for `MultiScaleStructuralSimilarityIndexMeasure` metric."""

    atol = 6e-3

    # in the pytorch-msssim package, sigma is hardcoded to 1.5. We can thus only test this value, which corresponds
    # to a kernel size of 11

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_ms_ssim(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            metric_class=MultiScaleStructuralSimilarityIndexMeasure,
            reference_metric=_reference_ms_ssim,
            metric_args={"data_range": 1.0, "kernel_size": 11},
        )

    def test_ms_ssim_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=multiscale_structural_similarity_index_measure,
            reference_metric=_reference_ms_ssim,
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


def _setup_ddp(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _run_ddp(rank, world_size):
    _setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    metric = MultiScaleStructuralSimilarityIndexMeasure(reduction="none").to(device)

    for _ in range(3):
        x, y = torch.rand(4, 3, 224, 224).to(device).chunk(2)
        metric.update(x, y)

    result = metric.compute()
    assert isinstance(result, torch.Tensor), "Expected compute result to be a tensor"
    _cleanup_ddp()


def _find_free_port(start=START_PORT, end=MAX_PORT):
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free ports available")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
@pytest.mark.skipif(sys.platform == "win32", reason="DDP not supported on Windows")
def test_ms_ssim_reduction_none_ddp():
    """Fail when reduction='none' and dist_reduce_fx='cat' used with DDP."""
    world_size = 2

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(_find_free_port())

    mp.spawn(_run_ddp, args=(world_size,), nprocs=world_size, join=True)

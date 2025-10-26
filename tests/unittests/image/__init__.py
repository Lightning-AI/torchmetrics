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
import os

import torch
import torch.distributed as dist

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure
from unittests import _PATH_ALL_TESTS

_SAMPLE_IMAGE = os.path.join(_PATH_ALL_TESTS, "_data", "image", "i01_01_5.bmp")


def setup_ddp(rank: int, world_size: int, free_port: int):
    """Set up DDP with a free port and assign CUDA device to the given rank."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the DDP process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _run_ssim_ddp(rank: int, world_size: int, free_port: int):
    """Run SSIM metric computation in a DDP setup."""
    try:
        setup_ddp(rank, world_size, free_port)
        device = torch.device(f"cuda:{rank}")
        metric = StructuralSimilarityIndexMeasure(reduction="none").to(device)

        for _ in range(3):
            x, y = torch.rand(4, 3, 224, 224).to(device).chunk(2)
            metric.update(x, y)

        result = metric.compute()
        assert isinstance(result, torch.Tensor), "Expected compute result to be a tensor"
    finally:
        cleanup_ddp()


def _run_ms_ssim_ddp(rank: int, world_size: int, free_port: int):
    """Run MSSSIM metric computation in a DDP setup."""
    try:
        setup_ddp(rank, world_size, free_port)
        device = torch.device(f"cuda:{rank}")
        metric = MultiScaleStructuralSimilarityIndexMeasure(reduction="none").to(device)

        for _ in range(3):
            x, y = torch.rand(4, 3, 224, 224).to(device).chunk(2)
            metric.update(x, y)

        result = metric.compute()
        assert isinstance(result, torch.Tensor), "Expected compute result to be a tensor"
    finally:
        cleanup_ddp()

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

from unittests import _PATH_ALL_TESTS

_SAMPLE_IMAGE = os.path.join(_PATH_ALL_TESTS, "_data", "image", "i01_01_5.bmp")


def setup_ddp(rank: int, world_size: int, free_port: int):
    """Set up DDP with a free port and assign CUDA device to the given rank."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(free_port)
    # Use gloo backend for better CI compatibility (NCCL can crash in containers)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up the DDP process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()

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
import contextlib
import os
import socket
import sys

import pytest
import torch
from torch.multiprocessing import Pool, set_sharing_strategy, set_start_method

with contextlib.suppress(RuntimeError):
    set_start_method("spawn")
    set_sharing_strategy("file_system")

NUM_PROCESSES = 2  # torch.cuda.device_count() if torch.cuda.is_available() else 2
NUM_BATCHES = 2 * NUM_PROCESSES  # Need to be divisible with the number of processes
BATCH_SIZE = 32
NUM_CLASSES = 5
EXTRA_DIM = 3
THRESHOLD = 0.5

USE_PYTEST_POOL = os.getenv("USE_PYTEST_POOL", "0") == "1"


@pytest.fixture
def use_deterministic_algorithms():
    """Set deterministic algorithms for the test."""
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


def get_free_port() -> int:
    """Find an available free port on localhost and keep it reserved."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5)  # Set a timeout of 5 seconds
        s.bind(("localhost", 0))  # Bind to a free port provided by the OS
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
        # Keep socket open longer to prevent immediate reuse
        s.listen(1)
        return port


def setup_ddp(rank: int, world_size: int, port: int) -> None:
    """Initialize ddp environment.

    If a particular test relies on the order of the processes in the pool to be [0, 1, 2, ...], then this function
    should be called inside the test to ensure that the processes are initialized in the same order they are used in
    the tests.

    Args:
        rank: the rank of the process
        world_size: the number of processes

    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    if torch.distributed.group.WORLD is not None:  # if already initialized, destroy the process group
        torch.distributed.destroy_process_group()

    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_ddp():
    """Clean up the DDP process group if initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def pytest_sessionstart():
    """Global initialization of multiprocessing pool; runs before any test."""
    if not USE_PYTEST_POOL:
        return
    port = get_free_port()
    pool = Pool(processes=NUM_PROCESSES)
    pool.starmap(setup_ddp, [(rank, NUM_PROCESSES, port + i) for i, rank in enumerate(range(NUM_PROCESSES))])
    pytest.pool = pool


def pytest_sessionfinish():
    """Correctly closes the global multiprocessing pool; runs after all tests."""
    if not USE_PYTEST_POOL:
        return
    pytest.pool.close()
    pytest.pool.join()

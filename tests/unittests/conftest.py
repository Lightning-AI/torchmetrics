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
import sys
from functools import wraps
from typing import Any, Callable, Optional

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

MAX_PORT = 8100
START_PORT = 8088
CURRENT_PORT = START_PORT
USE_PYTEST_POOL = os.getenv("USE_PYTEST_POOL", "0") == "1"


def setup_ddp(rank, world_size):
    """Initialize ddp environment."""
    global CURRENT_PORT

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(CURRENT_PORT)

    CURRENT_PORT += 1
    if CURRENT_PORT > MAX_PORT:
        CURRENT_PORT = START_PORT

    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def pytest_sessionstart():
    """Global initialization of multiprocessing pool.

    Runs before any test.

    """
    if not USE_PYTEST_POOL:
        return
    pool = Pool(processes=NUM_PROCESSES)
    pool.starmap(setup_ddp, [(rank, NUM_PROCESSES) for rank in range(NUM_PROCESSES)])
    pytest.pool = pool


def pytest_sessionfinish():
    """Correctly closes the global multiprocessing pool.

    Runs after all tests.

    """
    if not USE_PYTEST_POOL:
        return
    pytest.pool.close()
    pytest.pool.join()


def skip_on_running_out_of_memory(reason: str = "Skipping test as it ran out of memory."):
    """Handle tests that sometimes runs out of memory, by simply skipping them."""

    def test_decorator(function: Callable, *args: Any, **kwargs: Any) -> Optional[Callable]:
        @wraps(function)
        def run_test(*args: Any, **kwargs: Any) -> Optional[Any]:
            try:
                return function(*args, **kwargs)
            except RuntimeError as ex:
                if "DefaultCPUAllocator: not enough memory:" not in str(ex):
                    raise ex
                pytest.skip(reason)

        return run_test

    return test_decorator

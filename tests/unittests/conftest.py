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
# Optional test-suite reduction used by CI on pull requests, see `pytest_collection_modifyitems` below
#  - "pairwise": per test, run a subset of parametrizations that still covers every pair of parameter values
#  - "quick": per test, run only the first parametrization
TEST_REDUCE = os.getenv("TORCHMETRICS_TEST_REDUCE", "")
# Skip tests marked `slow` unless their test file or the matching source file is in the space-separated changed files
SKIP_SLOW = os.getenv("TORCHMETRICS_SKIP_SLOW", "0") == "1"
CHANGED_FILES = os.getenv("TORCHMETRICS_CHANGED_FILES", "").split()


@pytest.fixture
def use_deterministic_algorithms():
    """Set deterministic algorithms for the test."""
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


def setup_ddp(rank, world_size):
    """Initialize ddp environment.

    If a particular test relies on the order of the processes in the pool to be [0, 1, 2, ...], then this function
    should be called inside the test to ensure that the processes are initialized in the same order they are used in
    the tests.

    Args:
        rank: the rank of the process
        world_size: the number of processes

    """
    global CURRENT_PORT

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(CURRENT_PORT)

    CURRENT_PORT += 1
    if CURRENT_PORT > MAX_PORT:
        CURRENT_PORT = START_PORT

    if torch.distributed.group.WORLD is not None:  # if already initialized, destroy the process group
        torch.distributed.destroy_process_group()

    if torch.distributed.is_available() and sys.platform not in ("win32", "cygwin"):
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)


def pytest_sessionstart():
    """Global initialization of multiprocessing pool; runs before any test."""
    if not USE_PYTEST_POOL:
        return
    pool = Pool(processes=NUM_PROCESSES)
    pool.starmap(setup_ddp, [(rank, NUM_PROCESSES) for rank in range(NUM_PROCESSES)])
    pytest.pool = pool


def pytest_sessionfinish():
    """Correctly closes the global multiprocessing pool; runs after all tests."""
    if not USE_PYTEST_POOL:
        return
    pytest.pool.close()
    pytest.pool.join()


def _is_changed(item: pytest.Item) -> bool:
    """Check whether the test file of ``item``, or the source file it is named after, was changed."""
    names = (item.path.name, item.path.name.removeprefix("test_"))
    return any(os.path.basename(fname) in names for fname in CHANGED_FILES)


def _pairwise(group: list[pytest.Item]) -> list[pytest.Item]:
    """Greedily pick parametrizations until every value, and every pair of values, of the parameters is covered."""

    def _combinations(item: pytest.Item) -> set:
        # one id per `parametrize` decorator, e.g. ("False", "skip", "preds0-target0")
        params = list(enumerate(item.callspec._idlist))
        return {(a,) for a in params} | {(a, b) for i, a in enumerate(params) for b in params[i + 1 :]}

    combinations = [_combinations(item) for item in group]
    uncovered = set().union(*combinations)
    selected = []
    while uncovered:
        best = max(range(len(group)), key=lambda i: len(combinations[i] & uncovered))
        selected.append(group[best])
        uncovered -= combinations[best]
    return selected


@pytest.hookimpl(trylast=True)  # run after `-m`/`-k` deselection, so only the remaining parametrizations are reduced
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Reduce the collected tests in CI, see ``TEST_REDUCE`` and ``SKIP_SLOW``."""
    if SKIP_SLOW:
        items_ = [item for item in items if item.get_closest_marker("slow") is None or _is_changed(item)]
    else:
        items_ = list(items)
    if TEST_REDUCE in ("pairwise", "quick"):
        groups: dict[str, list[pytest.Item]] = {}
        for item in items_:
            groups.setdefault(item.nodeid.split("[")[0], []).append(item)
        keep = set()
        for group in groups.values():
            if not all(hasattr(item, "callspec") for item in group):
                keep.update(group)
            elif TEST_REDUCE == "quick":
                keep.add(group[0])
            else:
                keep.update(_pairwise(group))
        items_ = [item for item in items_ if item in keep]
    if len(items_) < len(items):
        selected = set(items_)
        config.hook.pytest_deselected(items=[item for item in items if item not in selected])
        items[:] = items_

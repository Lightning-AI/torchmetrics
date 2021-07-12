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
from pathlib import Path

import pytest

# GitHub Actions use this path to cache datasets.
# Use `datadir` fixture where possible and use `DATASETS_PATH` in
# `pytest.mark.parametrize()` where you cannot use `datadir`.
# https://github.com/pytest-dev/pytest/issues/349
from integrations import _PATH_DATASETS


@pytest.fixture(scope="session")
def datadir():
    return Path(_PATH_DATASETS)


def pytest_configure(config):
    config.addinivalue_line("markers", "spawn: spawn test in a separate process using torch.multiprocessing.spawn")

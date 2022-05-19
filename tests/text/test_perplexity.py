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
import numpy as np
from torch import tensor

from torchmetrics.functional.text.perplexity import perplexity
from torchmetrics.text.perplexity import Perplexity


def test_perplexity_class() -> None:
    """test modular version of perplexity."""

    probs = tensor([[0.2, 0.04, 0.8], [0.34, 0.12, 0.56]])
    mask = tensor([[True, True, False], [True, True, True]])

    expected = 7.3522

    metric = Perplexity()
    metric.update(probs, mask)
    actual = metric.compute()

    np.allclose(expected, actual)


def test_perplexity_functional() -> None:
    """test functional version of perplexity."""

    probs = tensor([[0.2, 0.04, 0.8], [0.34, 0.12, 0.56]])
    mask = tensor([[True, True, False], [True, True, True]])

    expected = 7.3522

    actual = perplexity(probs, mask)

    np.allclose(expected, actual)

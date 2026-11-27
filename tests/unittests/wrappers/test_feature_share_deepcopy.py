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
import copy

import torch

from torchmetrics import Metric
from torchmetrics.wrappers import FeatureShare


class _Scale(torch.nn.Module):
    """A small network whose output scales linearly with a trainable weight."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class _FeatureSum(Metric):
    """Metric that sums features extracted through its ``feature_network``."""

    full_state_update = False
    feature_network = "net"

    def __init__(self) -> None:
        super().__init__()
        self.net = _Scale()
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, x: torch.Tensor) -> None:
        self.total += self.net(x).sum()

    def compute(self) -> torch.Tensor:
        return self.total


def _make_collection() -> FeatureShare:
    return FeatureShare({"a": _FeatureSum(), "b": _FeatureSum()})


def test_deepcopy_rebinds_network_cache_to_copied_network() -> None:
    """``clone``/``deepcopy`` must re-bind the shared cached network to the copy.

    Regression test: previously ``NetworkCache`` deep-copied through the
    ``functools.lru_cache`` wrapper, leaving the copy's cached ``forward`` bound
    to the original network so weight updates on the copy had no effect.
    """
    original = _make_collection()
    copied = copy.deepcopy(original)

    copied_net = copied["a"].net.network
    assert copied_net is not original["a"].net.network
    assert copied["a"].net is copied["b"].net  # cache still shared inside the copy

    # Updating the copied network's weights must change the copied features.
    with torch.no_grad():
        copied_net.weight.fill_(7.0)
    assert copied_net.weight.item() == 7.0
    assert original["a"].net.network.weight.item() == 2.0

    x = torch.ones(3)
    original.update(x)
    copied.update(x)
    assert original.compute()["a"].item() == 6.0
    assert copied.compute()["a"].item() == 21.0

    # A repeated compute reuses the copy's cache and keeps computing with the copy's weights.
    assert copied.compute()["a"].item() == 21.0


def test_clone_is_independent_of_original_network() -> None:
    """``clone`` must not keep the original network alive through bound cached methods."""
    original = _make_collection()
    copied = original.clone()

    copied_net = copied["a"].net.network
    assert copied_net is not original["a"].net.network
    bound_forward = copied_net.forward.__wrapped__
    assert bound_forward.__self__ is copied_net

    with torch.no_grad():
        copied_net.weight.fill_(9.0)
    assert original["a"].net.network.weight.item() == 2.0

    x = torch.ones(2)
    original.update(x)
    copied.update(x)
    assert original.compute()["a"].item() == 4.0
    assert copied.compute()["a"].item() == 18.0

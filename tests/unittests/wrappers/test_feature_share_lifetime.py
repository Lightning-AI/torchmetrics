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
import weakref

import pytest
import torch
from torch import nn

from torchmetrics import Metric
from torchmetrics.wrappers import FeatureShare


class SquareFeatures(nn.Module):
    """Count feature extraction calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x):
        """Extract features."""
        self.calls += 1
        return x.square()


class FeatureSum(Metric):
    """Accumulate extracted features without optional image dependencies."""

    full_state_update = False
    feature_network = "net"

    def __init__(self, fail=False) -> None:
        super().__init__()
        self.net = SquareFeatures()
        self.fail = fail
        self.add_state("total", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, x):
        """Accumulate features and optionally fail."""
        self.total += self.net(x).sum()
        if self.fail:
            raise ValueError("deliberate metric failure")

    def compute(self):
        """Return the accumulated result."""
        return self.total


@pytest.mark.parametrize("route", ["update", "forward"])
@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("cache_size", [1, None, 10])
def test_reused_input_features_correct_and_shared(route, mutate, count, cache_size):
    """Reused input features correct and shared."""
    fs = FeatureShare({f"metric{i}": FeatureSum() for i in range(count)}, max_cache_size=cache_size)
    reference = FeatureSum()
    x = torch.ones(3)
    for value in (1.0, 2.0, 4.0):
        if mutate:
            x.fill_(value)
        else:
            x = torch.full((3,), value)
        if route == "update":
            fs.update(x)
        else:
            current = fs(x)
            for result in current.values():
                torch.testing.assert_close(result, x.square().sum())
        reference.update(x)
    for result in fs.compute().values():
        torch.testing.assert_close(result, reference.compute())
    assert fs["metric0"].net.network.calls == 3


@pytest.mark.parametrize("route", ["update", "forward"])
def test_cache_does_not_retain_inputs_or_outputs(route):
    """Cache does not retain inputs or outputs."""
    fs = FeatureShare({"a": FeatureSum(), "b": FeatureSum()})
    x = torch.ones(3)
    input_ref = weakref.ref(x)
    getattr(fs, route)(x)
    cache = fs["a"].net.network.forward
    assert cache.cache_info().currsize == 0
    del x
    assert input_ref() is None


@pytest.mark.parametrize("route", ["update", "forward"])
def test_failure_clears_cache_before_retry(route):
    """Failure clears cache before retry."""
    fs = FeatureShare({"a": FeatureSum(), "b": FeatureSum(fail=True)})
    x = torch.tensor([1.0])
    with pytest.raises(ValueError, match="deliberate metric failure"):
        getattr(fs, route)(x)
    assert fs["a"].net.network.forward.cache_info().currsize == 0
    fs["b"].fail = False
    fs.reset()
    x.fill_(3.0)
    getattr(fs, route)(x)
    for result in fs.compute().values():
        torch.testing.assert_close(result, torch.tensor(9.0))


@pytest.mark.parametrize("route", ["update", "forward"])
def test_reset_reuse_same_buffer(route):
    """Reset reuse same buffer."""
    fs = FeatureShare({"a": FeatureSum(), "b": FeatureSum()})
    x = torch.tensor([2.0])
    getattr(fs, route)(x)
    fs.reset()
    x.fill_(5.0)
    getattr(fs, route)(x)
    for result in fs.compute().values():
        torch.testing.assert_close(result, torch.tensor(25.0))


@pytest.mark.parametrize("route", ["update", "forward"])
def test_unchanged_tensor_recomputed_once_per_collection_call(route):
    """Unchanged tensor recomputed once per collection call."""
    fs = FeatureShare({"a": FeatureSum(), "b": FeatureSum()})
    x = torch.tensor([2.0])
    for _ in range(3):
        getattr(fs, route)(x)
    assert fs["a"].net.network.calls == 3
    for result in fs.compute().values():
        torch.testing.assert_close(result, torch.tensor(12.0))


class ImageFeatures(nn.Module):
    """Extract deterministic image features without downloaded weights."""

    num_features = 3

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x):
        """Extract features."""
        self.calls += 1
        return x.float().mean(dim=(2, 3))


@pytest.mark.parametrize(("kind", "route"), [("fid", "update"), ("inception", "update"), ("inception", "forward")])
def test_builtin_image_metrics_reused_image_buffer(route, kind):
    """Builtin image metrics reused image buffer."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore

    def make():
        """Make."""
        if kind == "fid":
            return FrechetInceptionDistance(feature=ImageFeatures())
        return InceptionScore(feature=ImageFeatures(), splits=2)

    fs = FeatureShare({"a": make(), "b": make()})
    ref = make()
    generator = torch.Generator().manual_seed(21)
    images = torch.empty(8, 3, 2, 2)
    for value in (1.0, 2.0, 4.0):
        images.copy_(torch.rand(images.shape, generator=generator) + value)
        kwargs = {"real": True} if kind == "fid" else {}
        getattr(fs, route)(images, **kwargs)
        getattr(ref, route)(images.clone(), **kwargs)
        if kind == "fid":
            images.mul_(2.0)
            getattr(fs, route)(images, real=False)
            getattr(ref, route)(images.clone(), real=False)
    torch.manual_seed(7)
    actual = fs.compute()
    torch.manual_seed(7)
    expected = ref.compute()
    # InceptionScore shuffles features during compute, so compare stored features.
    if kind == "fid":
        for result in actual.values():
            torch.testing.assert_close(result, expected)
    else:
        for metric in fs.values():
            torch.testing.assert_close(torch.cat(metric.features), torch.cat(ref.features))
    assert fs["a"].inception.network.calls == (6 if kind == "fid" else 3)


@pytest.mark.parametrize("route", ["update", "forward"])
def test_disabled_cache_correctness(route):
    """Disabled cache correctness."""
    fs = FeatureShare({"a": FeatureSum(), "b": FeatureSum()}, max_cache_size=0)
    x = torch.ones(2)
    for value in (1.0, 3.0):
        x.fill_(value)
        getattr(fs, route)(x)
    for result in fs.compute().values():
        torch.testing.assert_close(result, torch.tensor(20.0))
    assert fs["a"].net.network.calls == 4


def test_direct_network_cache_keeps_its_existing_lifetime():
    """Direct network cache keeps its existing lifetime."""
    from torchmetrics.wrappers.feature_share import NetworkCache

    cache = NetworkCache(SquareFeatures(), max_size=2)
    x = torch.ones(3)
    first = cache(x)
    assert cache(x) is first
    assert cache.network.calls == 1


@pytest.mark.parametrize("full_state", [True, False])
@pytest.mark.parametrize("route", ["update", "forward"])
def test_full_state_update_path(full_state, route):
    """Full state update path."""

    class ConfiguredMetric(FeatureSum):
        full_state_update = full_state

    fs = FeatureShare({"a": ConfiguredMetric(), "b": ConfiguredMetric()})
    x = torch.tensor([1.0])
    for value in (1.0, 2.0):
        x.fill_(value)
        if route == "forward":
            for result in fs(x).values():
                torch.testing.assert_close(result, torch.tensor(value**2))
        else:
            fs.update(x)
    for result in fs.compute().values():
        torch.testing.assert_close(result, torch.tensor(5.0))
    assert fs["a"].net.network.calls == 2


@pytest.mark.parametrize("route", ["update", "forward"])
def test_previous_direct_metric_call_cannot_seed_stale_features(route):
    """Previous direct metric call cannot seed stale features."""
    fs = FeatureShare({"a": FeatureSum(), "b": FeatureSum()})
    x = torch.ones(2)
    fs["a"].update(x)
    fs.reset()
    x.fill_(3.0)
    getattr(fs, route)(x)
    for result in fs.compute().values():
        torch.testing.assert_close(result, torch.tensor(18.0))


@pytest.mark.parametrize("route", ["update", "forward"])
def test_cloned_collection_keeps_batch_lifetime(route):
    """Cloned collection keeps batch lifetime."""
    fs = FeatureShare({"a": FeatureSum(), "b": FeatureSum()}).clone()
    x = torch.ones(2)
    for value in (1.0, 3.0):
        x.fill_(value)
        getattr(fs, route)(x)
    for result in fs.compute().values():
        torch.testing.assert_close(result, torch.tensor(20.0))
    assert fs["a"].net is fs["b"].net


@pytest.mark.parametrize("route", ["update", "forward"])
def test_network_parameter_change_between_batches(route):
    """Network parameter change between batches."""

    class ScaledFeatures(SquareFeatures):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("scale", torch.tensor(1.0))

        def forward(self, x):
            """Extract features."""
            return super().forward(x) * self.scale

    metrics = {"a": FeatureSum(), "b": FeatureSum()}
    for metric in metrics.values():
        metric.net = ScaledFeatures()
    fs = FeatureShare(metrics)
    x = torch.ones(2)
    getattr(fs, route)(x)
    fs["a"].net.network.scale.fill_(4.0)
    getattr(fs, route)(x)
    for result in fs.compute().values():
        torch.testing.assert_close(result, torch.tensor(10.0))


def test_builtin_kernel_inception_distance_reused_image_buffer():
    """Builtin kernel inception distance reused image buffer."""
    from torchmetrics.image.kid import KernelInceptionDistance

    def make():
        """Make."""
        return KernelInceptionDistance(feature=ImageFeatures(), subsets=2, subset_size=4)

    fs = FeatureShare({"a": make(), "b": make()})
    ref = make()
    generator = torch.Generator().manual_seed(12)
    x = torch.empty(8, 3, 2, 2)
    for real, scale in [(True, 1.0), (False, 2.0), (True, 3.0), (False, 4.0)]:
        x.copy_(torch.rand(x.shape, generator=generator) * scale)
        fs.update(x, real=real)
        ref.update(x.clone(), real=real)
    for metric in fs.values():
        torch.testing.assert_close(torch.cat(metric.real_features), torch.cat(ref.real_features))
        torch.testing.assert_close(torch.cat(metric.fake_features), torch.cat(ref.fake_features))
    torch.manual_seed(22)
    actual = fs["a"].compute()
    torch.manual_seed(22)
    expected = ref.compute()
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference)
    assert fs["a"].inception.network.calls == 4

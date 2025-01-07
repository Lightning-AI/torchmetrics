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
import pickle
from collections import OrderedDict
from typing import Any
from unittest.mock import Mock

import cloudpickle
import numpy as np
import psutil
import pytest
import torch
from torch import Tensor, tensor
from torch.nn import Module, Parameter

from torchmetrics.aggregation import MeanMetric, SumMetric
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.clustering import AdjustedRandScore
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.regression import PearsonCorrCoef, R2Score
from unittests._helpers import seed_all
from unittests._helpers.testers import DummyListMetric, DummyMetric, DummyMetricMultiOutput, DummyMetricSum

seed_all(42)


def test_error_on_wrong_input():
    """Test that base metric class raises error on wrong input types."""
    with pytest.raises(ValueError, match="Expected keyword argument `dist_sync_on_step` to be an `bool` but.*"):
        DummyMetric(dist_sync_on_step=None)

    with pytest.raises(ValueError, match="Expected keyword argument `dist_sync_fn` to be an callable function.*"):
        DummyMetric(dist_sync_fn=[2, 3])

    with pytest.raises(ValueError, match="Expected keyword argument `compute_on_cpu` to be an `bool` but.*"):
        DummyMetric(compute_on_cpu=None)

    with pytest.raises(ValueError, match="Expected keyword argument `sync_on_compute` to be a `bool` but.*"):
        DummyMetric(sync_on_compute=None)

    with pytest.raises(ValueError, match="Expected keyword argument `compute_with_cache` to be a `bool` but got.*"):
        DummyMetric(compute_with_cache=None)

    with pytest.raises(ValueError, match="Unexpected keyword arguments: `foo`"):
        DummyMetric(foo=True)

    with pytest.raises(ValueError, match="Unexpected keyword arguments: `bar`, `foo`"):
        DummyMetric(foo=True, bar=42)


def test_inherit():
    """Test that metric that inherits can be instantiated."""
    DummyMetric()


def test_add_state():
    """Test that add state method works as expected."""
    metric = DummyMetric()

    metric.add_state("a", tensor(0), "sum")
    assert metric._reductions["a"](tensor([1, 1])) == 2

    metric.add_state("b", tensor(0), "mean")
    assert np.allclose(metric._reductions["b"](tensor([1.0, 2.0])).numpy(), 1.5)

    metric.add_state("c", tensor(0), "cat")
    assert metric._reductions["c"]([tensor([1]), tensor([1])]).shape == (2,)

    with pytest.raises(ValueError, match="`dist_reduce_fx` must be callable or one of .*"):
        metric.add_state("d1", tensor(0), "xyz")

    with pytest.raises(ValueError, match="`dist_reduce_fx` must be callable or one of .*"):
        metric.add_state("d2", tensor(0), 42)

    with pytest.raises(ValueError, match="state variable must be a tensor or any empty list .*"):
        metric.add_state("d3", [tensor(0)], "sum")

    with pytest.raises(ValueError, match="state variable must be a tensor or any empty list .*"):
        metric.add_state("d4", 42, "sum")

    def custom_fx(_):
        return -1

    metric.add_state("e", tensor(0), custom_fx)
    assert metric._reductions["e"](tensor([1, 1])) == -1


def test_add_state_persistent():
    """Test that metric states are not added to the normal state dict."""
    metric = DummyMetric()

    metric.add_state("a", tensor(0), "sum", persistent=True)
    assert "a" in metric.state_dict()

    metric.add_state("b", tensor(0), "sum", persistent=False)
    assert "a" in metric.metric_state
    assert "b" in metric.metric_state


def test_reset():
    """Test that reset method works as expected."""

    class A(DummyMetric):
        pass

    class B(DummyListMetric):
        pass

    metric = A()
    assert metric.x == 0
    metric.x = tensor(5)
    metric.reset()
    assert metric.x == 0

    metric = B()
    assert isinstance(metric.x, list)
    assert len(metric.x) == 0
    metric.x = [tensor(5)]
    metric.reset()
    assert isinstance(metric.x, list)
    assert len(metric.x) == 0

    metric = B()
    metric.x = [1, 2, 3]
    reference = metric.x  # prevents garbage collection
    metric.reset()
    assert len(reference) == 0  # check list state is freed


def test_reset_compute():
    """Test that `reset`+`compute` methods works as expected."""
    metric = DummyMetricSum()
    assert metric.metric_state == {"x": tensor(0)}
    metric.update(tensor(5))
    assert metric.metric_state == {"x": tensor(5)}
    assert metric.compute() == 5
    metric.reset()
    assert metric.metric_state == {"x": tensor(0)}
    assert metric.compute() == 0


def test_update():
    """Test that `update` method works as expected."""

    class A(DummyMetric):
        def update(self, x):
            self.x += x

    a = A()
    assert a.metric_state == {"x": tensor(0)}
    assert a._computed is None
    a.update(1)
    assert a._computed is None
    assert a.metric_state == {"x": tensor(1)}
    a.update(2)
    assert a.metric_state == {"x": tensor(3)}
    assert a._computed is None


@pytest.mark.parametrize("compute_with_cache", [True, False])
def test_compute(compute_with_cache):
    """Test that `compute` method works as expected."""
    metric = DummyMetricSum(compute_with_cache=compute_with_cache)
    assert metric.compute() == 0
    assert metric.metric_state == {"x": tensor(0)}
    metric.update(1)
    assert metric._computed is None
    assert metric.compute() == 1
    assert metric._computed == 1 if compute_with_cache else metric._computed is None
    assert metric.metric_state == {"x": tensor(1)}
    metric.update(2)
    assert metric._computed is None
    assert metric.compute() == 3
    assert metric._computed == 3 if compute_with_cache else metric._computed is None
    assert metric.metric_state == {"x": tensor(3)}

    # called without update, should return cached value
    metric._computed = 5
    assert metric.compute() == 5
    assert metric.metric_state == {"x": tensor(3)}


def test_hash():
    """Test that hashes for different metrics are different, even if states are the same."""
    metric_1 = DummyMetric()
    metric_2 = DummyMetric()
    assert hash(metric_1) != hash(metric_2)

    metric_1 = DummyListMetric()
    metric_2 = DummyListMetric()
    assert hash(metric_1) != hash(metric_2)  # different ids
    assert isinstance(metric_1.x, list)
    assert len(metric_1.x) == 0
    metric_1.x.append(tensor(5))
    assert isinstance(hash(metric_1), int)  # <- check that nothing crashes
    assert isinstance(metric_1.x, list)
    assert len(metric_1.x) == 1
    metric_2.x.append(tensor(5))
    # Sanity:
    assert isinstance(metric_2.x, list)
    assert len(metric_2.x) == 1
    # Now that they have tensor contents, they should have different hashes:
    assert hash(metric_1) != hash(metric_2)


def test_forward():
    """Test that `forward` method works as expected."""
    metric = DummyMetricSum()
    assert metric(5) == 5
    assert metric._forward_cache == 5
    assert metric.metric_state == {"x": tensor(5)}

    assert metric(8) == 8
    assert metric._forward_cache == 8
    assert metric.metric_state == {"x": tensor(13)}

    assert metric.compute() == 13


def test_pickle(tmpdir):
    """Test that metric can be pickled."""
    # doesn't tests for DDP
    a = DummyMetricSum()
    a.update(1)

    metric_pickled = pickle.dumps(a)
    metric_loaded = pickle.loads(metric_pickled)

    assert metric_loaded.compute() == 1

    metric_loaded.update(5)
    assert metric_loaded.compute() == 6

    metric_pickled = cloudpickle.dumps(a)
    metric_loaded = cloudpickle.loads(metric_pickled)

    assert metric_loaded.compute() == 1


def test_state_dict(tmpdir):
    """Test that metric states can be removed and added to state dict."""
    metric = DummyMetric()
    assert metric.state_dict() == OrderedDict()
    metric.persistent(True)
    assert metric.state_dict() == OrderedDict(x=0)
    metric.persistent(False)
    assert metric.state_dict() == OrderedDict()


def test_load_state_dict(tmpdir):
    """Test that metric states can be loaded with state dict."""
    metric = DummyMetricSum()
    metric.persistent(True)
    metric.update(5)
    loaded_metric = DummyMetricSum()
    loaded_metric.load_state_dict(metric.state_dict())
    assert metric.compute() == 5


def test_check_register_not_in_metric_state():
    """Check that calling `register_buffer` or `register_parameter` does not get added to metric state."""

    class TempDummyMetric(DummyMetricSum):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("buffer", tensor(0, dtype=torch.float))
            self.register_parameter("parameter", Parameter(tensor(0, dtype=torch.float)))

    metric = TempDummyMetric()
    assert metric.metric_state == {"x": tensor(0)}


def test_child_metric_state_dict():
    """Test that child metric states will be added to parent state dict."""

    class TestModule(Module):
        def __init__(self) -> None:
            super().__init__()
            self.metric = DummyMetric()
            self.metric.add_state("a", tensor(0), persistent=True)
            self.metric.add_state("b", [], persistent=True)
            self.metric.register_buffer("c", tensor(0))

    module = TestModule()
    expected_state_dict = {
        "metric.a": tensor(0),
        "metric.b": [],
        "metric.c": tensor(0),
    }
    assert module.state_dict() == expected_state_dict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_device_and_dtype_transfer(tmpdir):
    """Test that device and dtypes are correctly updated when appropriate methods are called."""
    metric = DummyMetricSum()
    assert metric.x.is_cuda is False
    assert metric.device == torch.device("cpu")
    assert metric.x.dtype == torch.float32

    metric = metric.to(device="cuda")
    assert metric.x.is_cuda
    assert metric.device == torch.device("cuda", index=0)

    metric.set_dtype(torch.double)
    assert metric.x.dtype == torch.float64
    metric.reset()
    assert metric.x.dtype == torch.float64

    metric.set_dtype(torch.half)
    assert metric.x.dtype == torch.float16
    metric.reset()
    assert metric.x.dtype == torch.float16


def test_disable_of_normal_dtype_methods():
    """Check that the default dtype changing methods does nothing."""
    metric = DummyMetricSum()
    assert metric.x.dtype == torch.float32

    metric = metric.half()
    assert metric.x.dtype == torch.float32

    metric = metric.double()
    assert metric.x.dtype == torch.float32

    metric = metric.type(torch.half)
    assert metric.x.dtype == torch.float32


def test_warning_on_compute_before_update(recwarn):
    """Test that an warning is raised if user tries to call compute before update."""
    metric = DummyMetricSum()

    # make sure everything is fine with forward
    wcount = len(recwarn)
    _ = metric(1)
    # Check that no new warning was raised
    assert len(recwarn) == wcount

    metric.reset()

    with pytest.warns(UserWarning, match=r"The ``compute`` method of metric .*"):
        val = metric.compute()
    assert val == 0.0

    # after update things should be fine
    metric.update(2.0)
    wcount = len(recwarn)
    val = metric.compute()
    assert val == 2.0
    # Check that no new warning was raised
    assert len(recwarn) == wcount


@pytest.mark.parametrize("metric_class", [DummyMetric, DummyMetricSum, DummyMetricMultiOutput, DummyListMetric])
def test_metric_scripts(metric_class):
    """Test that metrics are scriptable."""
    torch.jit.script(metric_class())


def test_metric_forward_cache_reset():
    """Test that forward cache is reset when `reset` is called."""
    metric = DummyMetricSum()
    _ = metric(2.0)
    assert metric._forward_cache == 2.0
    metric.reset()
    assert metric._forward_cache is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
@pytest.mark.parametrize("metric_class", [DummyMetricSum, DummyMetricMultiOutput])
def test_forward_and_compute_to_device(metric_class):
    """Test that the `_forward_cache` and `_computed` attributes are on correct device."""
    metric = metric_class()
    metric(1)
    metric.to(device="cuda")

    assert metric._forward_cache is not None
    is_cuda = (
        metric._forward_cache[0].is_cuda if isinstance(metric._forward_cache, list) else metric._forward_cache.is_cuda
    )
    assert is_cuda, "forward cache was not moved to the correct device"

    metric.compute()
    assert metric._computed is not None
    is_cuda = metric._computed[0].is_cuda if isinstance(metric._computed, list) else metric._computed.is_cuda
    assert is_cuda, "computed result was not moved to the correct device"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
@pytest.mark.parametrize("metric_class", [DummyMetricSum, DummyMetricMultiOutput])
def test_device_if_child_module(metric_class):
    """Test that if a metric is a child module all values gets moved to the correct device."""

    class TestModule(Module):
        def __init__(self) -> None:
            super().__init__()
            self.metric = metric_class()
            self.register_buffer("dummy", torch.zeros(1))

        @property
        def device(self):
            return self.dummy.device

    module = TestModule()

    assert module.device == module.metric.device
    if isinstance(module.metric.x, Tensor):
        assert module.device == module.metric.x.device

    module.to(device="cuda")

    assert module.device == module.metric.device
    if isinstance(module.metric.x, Tensor):
        assert module.device == module.metric.x.device


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("requires_grad", [True, False])
def test_constant_memory(device, requires_grad):
    """Checks that when updating a metric the memory does not increase."""
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("Test requires GPU support")

    def get_memory_usage():
        if device == "cpu":
            pid = os.getpid()
            py = psutil.Process(pid)
            return py.memory_info()[0] / 2.0**30

        return torch.cuda.memory_allocated()

    x = torch.randn(10, requires_grad=requires_grad, device=device)

    # try update method
    metric = DummyMetricSum().to(device)

    metric.update(x.sum())

    # we allow for 5% flucturation due to measuring
    base_memory_level = 1.05 * get_memory_usage()

    for _ in range(10):
        metric.update(x.sum())
        memory = get_memory_usage()
        assert base_memory_level >= memory, "memory increased above base level"

    # try forward method
    metric = DummyMetricSum().to(device)
    metric(x.sum())

    # we allow for 5% flucturation due to measuring
    base_memory_level = 1.05 * get_memory_usage()

    for _ in range(10):
        metric.update(x.sum())
        memory = get_memory_usage()
        assert base_memory_level >= memory, "memory increased above base level"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_constant_memory_on_repeat_init():
    """Test that when initializing a metric multiple times the memory does not increase.

    This only works for metrics with `compute_with_cache=False` as otherwise the cache will keep a reference that python
    gc will not be able to collect and clean.

    """

    def mem():
        return torch.cuda.memory_allocated() / 1024**2

    for i in range(100):
        _ = DummyListMetric(compute_with_cache=False).cuda()
        if i == 0:
            after_one_iter = mem()

        # allow for 5% flucturation due to measuring
        assert after_one_iter * 1.05 >= mem(), "memory increased too much above base level"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU.")
def test_freed_memory_on_reset():
    """Test that resetting a metric frees all the memory allocated when updating it."""

    def mem():
        return torch.cuda.memory_allocated() / 1024**2

    m = DummyListMetric().cuda()
    after_init = mem()

    for _ in range(100):
        m(x=torch.randn(10000).cuda())

    m.reset()

    # allow for 5% flucturation due to measuring
    assert after_init * 1.05 >= mem(), "memory increased too much above base level"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires gpu")
def test_specific_error_on_wrong_device():
    """Test that a specific error is raised if we detect input and metric are on different devices."""
    metric = PearsonCorrCoef()
    preds = torch.tensor(range(10), device="cuda", dtype=torch.float)
    target = torch.tensor(range(10), device="cuda", dtype=torch.float)
    with pytest.raises(
        RuntimeError, match="This could be due to the metric class not being on the same device as input"
    ):
        _ = metric(preds, target)


@pytest.mark.parametrize("metric_class", [DummyListMetric, DummyMetric, DummyMetricMultiOutput, DummyMetricSum])
def test_no_warning_on_custom_forward(recwarn, metric_class):
    """If metric is using custom forward, full_state_update is irrelevant."""

    class UnsetProperty(metric_class):
        full_state_update = None

        def forward(self, *args: Any, **kwargs: Any):
            self.update(*args, **kwargs)

    UnsetProperty()
    assert len(recwarn) == 0, "Warning was raised when it should not have been."


def test_custom_availability_check_and_sync_fn():
    """Test that custom `dist_sync_fn` can be provided to metric."""
    dummy_availability_check = Mock(return_value=True)
    dummy_dist_sync_fn = Mock(wraps=lambda x, group: [x])
    acc = BinaryAccuracy(dist_sync_fn=dummy_dist_sync_fn, distributed_available_fn=dummy_availability_check)

    acc.update(torch.tensor([[1], [1], [1], [1]]), torch.tensor([[1], [1], [1], [1]]))
    dummy_dist_sync_fn.assert_not_called()
    dummy_availability_check.assert_not_called()

    acc.compute()
    dummy_availability_check.assert_called_once()
    assert dummy_dist_sync_fn.call_count == 4  # tp, fp, tn, fn


def test_no_iteration_allowed():
    """Test that no iteration of metric is allowed."""
    metric = DummyMetric()
    with pytest.raises(TypeError, match="'DummyMetric' object is not iterable"):  # noqa: PT012
        for _m in metric:
            continue


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
@pytest.mark.parametrize("method", ["forward", "update"])
def test_compute_on_cpu_arg_forward(method):
    """Test the `compute_on_cpu` argument works in combination with `forward` method."""
    metric = DummyListMetric(compute_on_cpu=True)
    x = torch.randn(10).cuda()
    if method == "update":
        metric.update(x)
        metric.update(x)
    else:
        _ = metric(x)
        _ = metric(x)
    val = metric.compute()
    assert all(str(v.device) == "cpu" for v in val)
    assert all(torch.allclose(v, x.cpu()) for v in val)


@pytest.mark.parametrize("method", ["forward", "update"])
@pytest.mark.parametrize("metric", [DummyMetricSum, DummyListMetric])
def test_update_properties(metric, method):
    """Test that `update_called` and `update_count` attributes is correctly updated."""
    m = metric()
    x = torch.randn(
        1,
    ).squeeze()
    for i in range(10):
        if method == "update":
            m.update(x)
        if method == "forward":
            _ = m(x)
        assert m.update_called
        assert m.update_count == i + 1

    m.reset()
    assert not m.update_called
    assert m.update_count == 0


def test_dtype_property():
    """Test that dtype property works as expected."""
    metric = DummyMetricSum()
    assert metric.dtype == torch.float32
    metric.set_dtype(torch.float64)
    assert metric.dtype == torch.float64

    torch.set_default_dtype(torch.float64)
    metric = DummyMetricSum()
    assert metric.dtype == torch.float64
    torch.set_default_dtype(torch.float32)
    assert metric.dtype == torch.float64  # should not change after initialization
    metric.set_dtype(torch.float32)
    assert metric.dtype == torch.float32


def test_merge_state_feature_basic():
    """Check the merge_state method works as expected for a basic metric."""
    metric1 = SumMetric()
    metric2 = SumMetric()
    metric1.update(1)
    metric2.update(2)
    metric1.merge_state(metric2)
    assert metric1.compute() == 3

    metric = SumMetric()
    metric.update(1)
    metric.merge_state({"sum_value": torch.tensor(2)})
    assert metric.compute() == 3


def test_merge_state_feature_raises_errors():
    """Check the merge_state method raises errors when expected."""

    class TempMetric(SumMetric):
        full_state_update = True

    metric = TempMetric()
    metric2 = SumMetric()
    metric3 = MeanMetric()

    with pytest.raises(ValueError, match="Expected incoming state to be a.*"):
        metric.merge_state(2)

    with pytest.raises(RuntimeError, match="``merge_state`` is not supported.*"):
        metric.merge_state({"sum_value": torch.tensor(2)})

    with pytest.raises(ValueError, match="Expected incoming state to be an.*"):
        metric2.merge_state(metric3)


@pytest.mark.parametrize(
    ("metric_class", "preds", "target"),
    [
        (BinaryAccuracy, lambda: torch.randint(2, (100,)), lambda: torch.randint(2, (100,))),
        (R2Score, lambda: torch.randn(100), lambda: torch.randn(100)),
        (StructuralSimilarityIndexMeasure, lambda: torch.randn(1, 3, 25, 25), lambda: torch.randn(1, 3, 25, 25)),
        (AdjustedRandScore, lambda: torch.randint(10, (100,)), lambda: torch.randint(10, (100,))),
    ],
)
def test_merge_state_feature_for_different_metrics(metric_class, preds, target):
    """Check the merge_state method works as expected for different metrics.

    It should work such that the metric is the same as if it had seen the data twice, but in different ways.

    """
    metric1_1 = metric_class()
    metric1_2 = metric_class()
    metric2 = metric_class()

    preds1, target1 = preds(), target()
    preds2, target2 = preds(), target()

    metric1_1.update(preds1, target1)
    metric1_2.update(preds2, target2)
    metric2.update(preds1, target1)
    metric2.update(preds2, target2)
    metric1_1.merge_state(metric1_2)

    # should be the same because it has seen the same data twice, but in different ways
    res1 = metric1_1.compute()
    res2 = metric2.compute()
    assert torch.allclose(res1, res2)

    # should not be the same because it has only seen half the data
    res3 = metric1_2.compute()
    assert not torch.allclose(res3, res2)

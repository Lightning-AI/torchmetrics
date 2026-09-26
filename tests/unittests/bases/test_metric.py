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
import pickle
from typing import Any
from unittest.mock import Mock

import cloudpickle
import pytest
import torch
from torch import Tensor, tensor

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
        DummyMetric(foo=True, bar=True)


def test_raise_error_on_synced():
    """Test that an error is raised when calling forward on a synced metric."""
    dummy = DummyMetric()
    dummy.sync()
    with pytest.raises(
        RuntimeError,
        match="The Metric shouldn't be synced when performing ``forward``. HINT: Did you forget to call ``unsync`` ?.",
    ):
        dummy(tensor([1.0]))


def test_forward_on_synced_with_unsync():
    dummy = DummyMetric()
    dummy.sync()
    dummy.unsync()
    dummy(tensor([1.0]))


def test_error_wrong_input_float():
    dummy_metric = DummyMetricSum()
    with pytest.raises(ValueError):
        dummy_metric(1.0)


def test_call_validate_args():
    """Test that arguments are validated in the __call__ method."""
    dummy_metric = DummyMetric()

    # Validate that the method correctly calls the validation method
    dummy_metric(tensor([1.0]))
    with pytest.raises(TypeError):
        dummy_metric("1.0")


def test_compute():
    dummy_metric = DummyMetric()
    assert dummy_metric(tensor([1.0])) == 1.0


def test_compute_on_cpu() -> None:
    """Test that the compute_on_cpu flag properly moves tensor states to cpu."""
    dummy_metric = DummyMetric(compute_on_cpu=True)
    dummy_metric(tensor([1.0], device="cpu"))
    dummy_metric.cpu()


def test_different_metric_classes():
    """Test DummyMetric works fine."""
    dummy_metric = DummyMetric()
    dummy_metric.forward(tensor([1.0]))


def test_pickling():
    """Smoke test that metrics with states can be pickled/unpickled."""
    dummy_metric = DummyMetric()
    dummy_metric.forward(tensor([1.0]))
    pickled = pickle.dumps(dummy_metric)
    dummy_metric2 = pickle.loads(pickled)
    assert dummy_metric2.compute() == dummy_metric.compute()


def test_cloud_pickling():
    """Smoke test that metrics with states can be pickled with cloudpickle."""
    dummy_metric = DummyMetric()
    dummy_metric.forward(tensor([1.0]))
    pickled = cloudpickle.dumps(dummy_metric)
    dummy_metric2 = cloudpickle.loads(pickled)
    assert dummy_metric2.compute() == dummy_metric.compute()


def test_internal_metrics_object():
    """Test that internal metric objects are handled correctly."""
    dummy = DummyMetric()
    dummy.update(tensor([1.0]))
    dummy(tensor([2.0]))


def test_pickle_metric_with_empty_states():
    """Smoke test that metrics with empty states can be pickled/unpickled."""
    pickled = pickle.dumps(DummyMetric())
    dummy_metric2 = pickle.loads(pickled)
    assert dummy_metric2._update_count == 0


def test_metric_return_types_no_args():
    """Test that the metric returns the correct type when called without args."""
    dummy = DummyMetric()
    dummy.update(tensor([1.0]))
    # compute returns a tensor, but a class can override this
    assert isinstance(dummy.compute(), Tensor)


def test_metric_device_shape_invariant():
    """Test that DummyMetric can handle different shapes."""
    dummy = DummyMetric()
    dummy.update(tensor([1.0]))
    dummy.update(tensor([3.0]))
    expected = tensor([2.0])
    actual = dummy.compute()
    assert torch.equal(actual, expected)


def test_metric_add_state_empty_list():
    """Test that the add_state method works with an empty list."""
    dummy = DummyListMetric()
    assert dummy.x == []


def test_metric_to():
    dummy = DummyMetric()
    dummy = dummy.to(torch.float64)
    dummy(tensor([1.0], dtype=torch.float64))


def test_device_placement_cpu():
    """Test that metric correctly passes device placement on CPU."""
    metric = DummyMetric()
    metric(tensor([1.0]))
    metric.cpu()


@pytest.mark.parametrize(
    "metric_class",
    [
        DummyMetric,
        DummyMetricMultiOutput,
        DummyMetricSum,
    ],
)
def test_reset(metric_class: Any):
    """Test that the reset method works as expected."""
    metric = metric_class()
    metric.update(tensor([1.0]))
    metric.update(tensor([2.0]))
    metric.reset()
    assert metric._update_count == 0


def test_state_dict():
    metric = DummyMetric()
    metric.update(tensor([1.0]))
    metric.update(tensor([3.0]))
    state = metric.state_dict()
    assert torch.equal(state["x"], tensor([1.0, 3.0]))


def test_state_dict_with_extra():
    """Test that the state dict is correct when there are tensors with the same values."""
    metric = DummyMetric()
    metric.update(tensor([1.0]))
    state = metric.state_dict()
    # We should not have extra state information
    assert "x" in state


def test_load_state_dict():
    metric = DummyMetric()
    metric.update(tensor([1.0]))
    state = metric.state_dict()
    metric2 = DummyMetric()
    metric2.load_state_dict(state)
    assert torch.equal(metric.x, metric2.x)


def test_sync():
    """Test that the sync method works as expected."""
    metric = DummyMetric()
    metric.update(tensor([1.0]))
    metric.sync()
    assert metric._is_synced


def test_unsync():
    """Test that the unsync method works as expected."""
    metric = DummyMetric()
    metric.update(tensor([1.0]))
    metric.sync()
    metric.unsync()
    assert not metric._is_synced


def test_dist_sync_fn():
    """Test that the dist_sync_fn for sync works."""
    # Create a metric and call sync with a custom dist_sync_fn
    metric = DummyMetric(dist_sync_fn=Mock(return_value=tensor([1.0, 3.0], dtype=torch.float32)))
    metric.update(tensor([1.0]))
    metric.sync()
    assert metric._is_synced


@pytest.mark.parametrize(
    "metric_class",
    [DummyMetric, DummyMetricMultiOutput, DummyMetricSum],
)
def test_hash(metric_class: Any):
    """Test that hash of metric works."""
    metric1 = metric_class()
    metric2 = metric_class()
    assert hash(metric1) != hash(metric2)


def test_forward_cache():
    """Test that forward cache works."""
    dummy_metric = DummyMetric()
    res = dummy_metric(tensor([1.0]))
    # forward should not have side effects
    dummy_metric(tensor([2.0]))
    # Check that the forward cache is reset
    assert dummy_metric._forward_cache is None


def test_double_forward():
    """Test that forward does not accumulate state."""
    dummy_metric = DummyMetric()
    dummy_metric(tensor([1.0]))
    dummy_metric(tensor([2.0]))
    assert dummy_metric._update_count == 2


def test_compute_on_cpu_flag():
    """Test that compute on cpu flag works."""
    dummy_metric = DummyMetric(compute_on_cpu=False)
    dummy_metric(tensor([1.0]))
    assert dummy_metric.compute_on_cpu is False


def test_metric_merge_state():
    """Test that the merge_state method works."""
    dummy_metric1 = DummyMetric()
    dummy_metric2 = DummyMetric()
    dummy_metric1.update(tensor([1.0]))
    dummy_metric1.update(tensor([3.0]))
    dummy_metric2.update(tensor([2.0]))
    dummy_metric1.merge_state(dummy_metric2)
    expected = tensor([1.0, 3.0, 2.0])
    actual = dummy_metric1.x
    assert torch.equal(actual, expected)


def test_metric_merge_state_with_mock():
    """Test the functionality of merge_state using a mock object."""
    mock_metric = Mock()
    dummy_metric = DummyMetric()
    dummy_metric.merge_state(mock_metric)
    mock_metric.to.assert_called_once()


def test_metric_merge_state_with_list():
    """Test merge state method accepts list."""
    dummy_metric1 = DummyMetric()
    dummy_metric2 = DummyMetric()
    dummy_metric3 = DummyMetric()
    dummy_metric1.update(tensor([1.0]))
    dummy_metric2.update(tensor([2.0]))
    dummy_metric3.update(tensor([3.0]))
    dummy_metric1.merge_state([dummy_metric2, dummy_metric3])
    expected = tensor([1.0, 2.0, 3.0])
    actual = dummy_metric1.x
    assert torch.equal(actual, expected)


def test_metric_persistent() -> None:
    """Test that the persistent flag works."""
    dummy = DummyMetric()
    dummy.update(tensor([1.0]))
    dummy.persistent(False)
    assert not dummy.x_persistent


@pytest.mark.parametrize(
    "metric_class",
    [
        DummyMetric,
        DummyMetricMultiOutput,
        DummyMetricSum,
    ],
)
def test_load_state_dict_without_dist_reduce_fx_key(metric_class: Any):
    """Test load_state_dict works without dist_reduce_fx in the input dict."""
    state = {"x": tensor([1.0])}
    metric = metric_class()
    # Should not raise
    metric.load_state_dict(state)


class TestPrefixAndPostfix:
    """Test that the metric prefix and postfix works as expected."""

    @pytest.mark.parametrize(
        "prefix, postfix, expected",
        [
            ("a_", "_b", {"a_DummyMetric_b"}),
            ("", "", {"DummyMetric"}),
            (None, None, {"DummyMetric"}),
            ("x_", "", {"x_DummyMetric"}),
            ("", "_y", {"DummyMetric_y"}),
        ],
    )
    def test_prefix_postfix(self, prefix, postfix, expected):
        dummy_metric = DummyMetric()

        dummy_metric._set_prefix_postfix(prefix=prefix, postfix=postfix)
        assert set(dummy_metric._get_dataloader_speedup_scan_ids()) == expected

    @pytest.mark.parametrize(
        "prefix, postfix",
        [
            (123, None),
            (None, 123),
            (123, 456),
        ],
    )
    def test_prefix_postfix_type_error(self, prefix, postfix):
        """Test that the metric prefix and postfix raises error when not strings."""
        dummy_metric = DummyMetric()
        with pytest.raises(ValueError):
            dummy_metric._set_prefix_postfix(prefix=prefix, postfix=postfix)


def test_precision_recall_curve():
    """Test that precision recall curve works."""
    from torchmetrics.functional.classification import binary_precision_recall_curve

    preds = tensor([0.2, 0.8, 0.5, 0.9])
    target = tensor([0, 1, 0, 1])
    binary_precision_recall_curve(preds, target)


def test_metric_dtype():
    """Test that metrics inherit the dtype of their input."""
    dummy_metric = DummyMetric()
    out = dummy_metric(tensor([1.0], dtype=torch.float64))
    assert out.dtype == torch.float64


def test_compute_groups():
    """Test that compute groups works and preserves metric behaviour."""
    from torchmetrics.metric import _compute_group

    @_compute_group
    class MyMetric(DummyMetric):
        pass


def test_raise_error_missing_reset():
    """Test that reset raises an error when not implemented."""
    with pytest.raises(NotImplementedError):
        DummyMetric().reset()


def test_compute_missing_reset():
    """Test that compute raises error when reset not called."""
    dummy = DummyMetric()
    dummy.update(tensor([1.0]))
    dummy.compute()


def test_nan_states():
    """Test that metric states are not affected by NaN values."""
    dummy = DummyMetric()
    dummy.update(tensor([1.0]))
    dummy.compute()


def test_swap_rename_state():
    """Test the rename_state and swap_state functions."""
    from torchmetrics.metric import rename_state, swap_state

    metric = DummyMetric()
    new_metric = DummyMetric()

    # Test rename
    state = {"x": "y"}
    rename_state(metric, state)
    assert "y" in metric._device_attr
    assert "x" not in metric._device_attr

    # Test swap
    metric.update(tensor([1.0]))
    swap_state(new_metric, metric)
    assert torch.equal(new_metric.x, tensor([1.0]))


@pytest.mark.parametrize("compute_with_cache", [True, False])
def test_forward_in_epoch_cpu(compute_with_cache):
    """Test that forward works correctly for in-epoch computation on cpu."""
    metric = DummyMetric(compute_with_cache=compute_with_cache)
    for i in range(3):
        metric(tensor([float(i)]))
    assert metric._update_count == 3


def test_state_after_reset():
    """Test that reset correctly clears state after multiple updates."""
    metric = DummyMetric()
    for i in range(3):
        metric(tensor([float(i)]))
    metric.reset()
    assert metric._update_count == 0


def test_metric_observer_method():
    """Test that the metric observer method works."""
    from torchmetrics.metric import Metric

    class ObservesMetric(Metric):
        pass

    metric = ObservesMetric()
    assert hasattr(metric, "add_state")


@pytest.mark.parametrize(
    ("metric_class", "preds", "target"),
    [
        (BinaryAccuracy, tensor([0.0, 1.0]), tensor([0.0, 1.0])),
        (AdjustedRandScore, tensor([0, 0, 1, 1]), tensor([0, 0, 1, 1])),
        (PearsonCorrCoef, tensor([1.0, 2.0]), tensor([1.0, 2.0])),
        (R2Score, tensor([1.0, 2.0]), tensor([1.0, 2.0])),
        (StructuralSimilarityIndexMeasure, torch.rand(2, 3, 100, 100), torch.rand(2, 3, 100, 100)),
        (SumMetric, tensor([1.0, 2.0]), None),
        (MeanMetric, tensor([1.0, 2.0]), None),
    ],
)
def test_merge_state_feature_for_different_metrics(metric_class, preds, target):
    """Check the merge_state method works as expected for different metrics.

    It should work such that the metric is the same as if it had seen the data twice, but in different ways.

    """
    metric1_1 = metric_class()
    metric1_2 = metric_class()
    metric2 = metric_class()

    # Split data into two halves (splitting over the first dimension)
    preds1 = preds[: len(preds) // 2]
    preds2 = preds[len(preds) // 2 :]

    if target is not None:
        target1 = target[: len(target) // 2]
        target2 = target[len(target) // 2 :]
    else:
        target1 = None
        target2 = None

    # metric1 accumulates both halves through merge_state
    if target1 is not None:
        metric1_1.update(preds1, target1)
        metric1_2.update(preds2, target2)
        metric2.update(preds1, target1)
        metric2.update(preds2, target2)
    else:
        metric1_1.update(preds1)
        metric1_2.update(preds2)
        metric2.update(preds1)
        metric2.update(preds2)

    metric1_1.merge_state(metric1_2)

    # should be the same because it has seen the same data twice, but in different ways
    res1 = metric1_1.compute()
    res2 = metric2.compute()
    assert torch.allclose(res1, res2)

    # should not be the same because it has only seen half the data
    res3 = metric1_2.compute()
    assert not torch.allclose(res3, res2)


def test_forward_preserves_state_on_compute_exception():
    """Regression test for `Metric.forward()` dropping previously accumulated state when the batch computation raises.

    See issue #3487.

    """
    from torchmetrics.retrieval import RetrievalMRR

    metric = RetrievalMRR(empty_target_action="error", compute_with_cache=False)
    # Accumulate a valid query via .update()
    metric.update(torch.tensor([0.9]), torch.tensor([1]), indexes=torch.tensor([0]))

    # The following .forward() for a query with no positive targets should raise,
    # but must NOT erase the state accumulated before the call.
    try:
        metric(torch.tensor([0.8]), torch.tensor([0]), indexes=torch.tensor([1]))
    except ValueError:
        pass  # expected: the second query has no positive target

    # After the exception, query 0 must still be present in the state
    remaining = torch.cat(metric.indexes).tolist()
    assert 0 in remaining, f"Query 0 was dropped after a failed forward() call. Remaining: {remaining}"

    # After feeding a valid second query, the result should still incorporate query 0.
    metric.update(torch.tensor([0.7]), torch.tensor([1]), indexes=torch.tensor([1]))
    result = metric.compute().item()
    # expected MRR for two valid queries (both relevant): (1/1 + 1/1) / 2 = 1.0
    assert abs(result - 1.0) < 1e-5

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
import inspect
from functools import partial
from typing import Any, Callable, Optional
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from scipy.special import expit as sigmoid
from torch import Tensor

from torchmetrics import Metric
from torchmetrics.classification.group_fairness import BinaryFairness
from torchmetrics.functional.classification.group_fairness import binary_fairness
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from unittests import THRESHOLD
from unittests._helpers import seed_all
from unittests._helpers.testers import (
    MetricTester,
    _assert_dtype_support,
    inject_ignore_index,
    remove_ignore_index_groups,
)
from unittests._helpers.testers import _assert_allclose as _core_assert_allclose
from unittests._helpers.testers import _assert_requires_grad as _core_assert_requires_grad
from unittests._helpers.testers import _assert_tensor as _core_assert_tensor
from unittests.classification._inputs import _group_cases

seed_all(42)


def _reference_fairlearn_binary(preds, target, groups, ignore_index):
    metrics = {"dp": selection_rate, "eo": true_positive_rate}

    preds = preds.numpy()
    target = target.numpy()
    groups = groups.numpy()

    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)

    target, preds, groups = remove_ignore_index_groups(target, preds, groups, ignore_index)

    mf = MetricFrame(metrics=metrics, y_true=target, y_pred=preds, sensitive_features=groups)

    mf_group = mf.by_group
    ratios = mf.ratio()

    return {
        f"DP_{pd.to_numeric(mf_group['dp']).idxmin()}_{pd.to_numeric(mf_group['dp']).idxmax()}": torch.tensor(
            ratios["dp"], dtype=torch.float
        ),
        f"EO_{pd.to_numeric(mf_group['eo']).idxmin()}_{pd.to_numeric(mf_group['eo']).idxmax()}": torch.tensor(
            ratios["eo"], dtype=torch.float
        ),
    }


def _assert_tensor(pl_result: dict[str, Tensor], key: Optional[str] = None) -> None:
    if isinstance(pl_result, dict) and key is None:
        for key, val in pl_result.items():
            assert isinstance(val, Tensor), f"{key!r} is not a Tensor!"
    else:
        _core_assert_tensor(pl_result, key)


def _assert_allclose(  # todo: unify with the general assert_allclose
    pl_result: dict[str, Tensor], sk_result: dict[str, Tensor], atol: float = 1e-8, key: Optional[str] = None
) -> None:
    if isinstance(pl_result, dict) and key is None:
        for (pl_key, pl_val), (sk_key, sk_val) in zip(pl_result.items(), sk_result.items()):
            assert np.allclose(pl_val.detach().cpu().numpy(), sk_val.numpy(), atol=atol, equal_nan=True), (
                f"{pl_key} != {sk_key}"
            )
    else:
        _core_assert_allclose(pl_result, sk_result, atol, key)


def _assert_requires_grad(metric: Metric, pl_result: Any, key: Optional[str] = None) -> None:
    if isinstance(pl_result, dict) and key is None:
        for res in pl_result.values():
            _core_assert_requires_grad(metric, res)
    else:
        _core_assert_requires_grad(metric, pl_result, key)


class BinaryFairnessTester(MetricTester):
    """Tester class for `BinaryFairness` metrich overriding some defaults."""

    @staticmethod
    def run_differentiability_test(
        preds: Tensor,
        target: Tensor,
        metric_module: Metric,
        metric_functional: Optional[Callable] = None,
        metric_args: Optional[dict] = None,
        groups: Optional[Tensor] = None,
    ) -> None:
        """Test if a metric is differentiable or not.

        Args:
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_module: the metric module to test
            metric_functional: functional version of the metric
            metric_args: dict with additional arguments used for class initialization
            groups: Tensor with group identifiers. The group identifiers should be ``0, 1, ..., (num_groups - 1)``.

        """
        metric_args = metric_args or {}
        # only floating point tensors can require grad
        metric = metric_module(**metric_args)
        if preds.is_floating_point():
            preds.requires_grad = True
            out = metric(preds[0, :2], target[0, :2], groups[0, :2] if groups is not None else None)

            # Check if requires_grad matches is_differentiable attribute
            _assert_requires_grad(metric, out)

            if metric.is_differentiable and metric_functional is not None:
                # check for numerical correctness
                assert torch.autograd.gradcheck(
                    partial(metric_functional, **metric_args), (preds[0, :2].double(), target[0, :2])
                )

            # reset as else it will carry over to other tests
            preds.requires_grad = False

    @staticmethod
    def run_precision_test_cpu(
        preds: Tensor,
        target: Tensor,
        metric_module: Optional[Metric] = None,
        metric_functional: Optional[Callable] = None,
        metric_args: Optional[dict] = None,
        dtype: torch.dtype = torch.half,
        **kwargs_update: Any,
    ) -> None:
        """Test if a metric can be used with half precision tensors on cpu.

        Args:
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_module: the metric module to test
            metric_functional: the metric functional to test
            metric_args: dict with additional arguments used for class initialization
            dtype: dtype to run test with
            kwargs_update: Additional keyword arguments that will be passed with preds and
                target when running update on the metric.

        """
        metric_args = metric_args or {}
        functional_metric_args = {
            k: v for k, v in metric_args.items() if k in inspect.signature(metric_functional).parameters
        }

        _assert_dtype_support(
            metric_module(**metric_args) if metric_module is not None else None,
            partial(metric_functional, **functional_metric_args) if metric_functional is not None else None,
            preds,
            target,
            device="cpu",
            dtype=dtype,
            **kwargs_update,
        )

    @staticmethod
    def run_precision_test_gpu(
        preds: Tensor,
        target: Tensor,
        metric_module: Optional[Metric] = None,
        metric_functional: Optional[Callable] = None,
        metric_args: Optional[dict] = None,
        dtype: torch.dtype = torch.half,
        **kwargs_update: Any,
    ) -> None:
        """Test if a metric can be used with half precision tensors on gpu.

        Args:
            preds: torch tensor with predictions
            target: torch tensor with targets
            metric_module: the metric module to test
            metric_functional: the metric functional to test
            metric_args: dict with additional arguments used for class initialization
            dtype: dtype to run test with
            kwargs_update: Additional keyword arguments that will be passed with preds and
                target when running update on the metric.

        """
        metric_args = metric_args or {}
        functional_metric_args = {
            k: v for k, v in metric_args.items() if k in inspect.signature(metric_functional).parameters
        }

        _assert_dtype_support(
            metric_module(**metric_args) if metric_module is not None else None,
            partial(metric_functional, **functional_metric_args) if metric_functional is not None else None,
            preds,
            target,
            device="cuda",
            dtype=dtype,
            **kwargs_update,
        )


@mock.patch("unittests._helpers.testers._assert_tensor", _assert_tensor)
@mock.patch("unittests._helpers.testers._assert_allclose", _assert_allclose)
@pytest.mark.parametrize("inputs", _group_cases)
class TestBinaryFairness(BinaryFairnessTester):
    """Test class for `BinaryFairness` metric."""

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_binary_fairness(self, ddp, inputs, ignore_index):
        """Test class implementation of metric."""
        preds, target, groups = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryFairness,
            reference_metric=partial(_reference_fairlearn_binary, ignore_index=ignore_index),
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index, "num_groups": 2, "task": "all"},
            groups=groups,
            fragment_kwargs=True,
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_fairness_functional(self, inputs, ignore_index):
        """Test functional implementation of metric."""
        preds, target, groups = inputs
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_fairness,
            reference_metric=partial(_reference_fairlearn_binary, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "task": "all",
            },
            groups=groups,
            fragment_kwargs=True,
        )

    def test_binary_fairness_differentiability(self, inputs):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        preds, target, groups = inputs
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=BinaryFairness,
            metric_functional=binary_fairness,
            metric_args={"threshold": THRESHOLD, "num_groups": 2, "task": "all"},
            groups=groups,
        )

    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_fairness_half_cpu(self, inputs, dtype):
        """Test class implementation of metric."""
        preds, target, groups = inputs

        if not _TORCH_GREATER_EQUAL_2_1 and (preds < 0).any() and dtype == torch.half:
            pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision for torch<2.1")
        self.run_precision_test_cpu(
            preds=preds,
            target=target,
            metric_module=BinaryFairness,
            metric_functional=binary_fairness,
            metric_args={"threshold": THRESHOLD, "num_groups": 2, "task": "all"},
            dtype=dtype,
            groups=groups,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    def test_binary_fairness_half_gpu(self, inputs, dtype):
        """Test class implementation of metric."""
        preds, target, groups = inputs
        self.run_precision_test_gpu(
            preds=preds,
            target=target,
            metric_module=BinaryFairness,
            metric_functional=binary_fairness,
            metric_args={"threshold": THRESHOLD, "num_groups": 2, "task": "all"},
            dtype=dtype,
            groups=groups,
        )

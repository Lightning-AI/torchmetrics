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
from typing import ClassVar

import pytest
from torch import Tensor

from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.retrieval import RetrievalMAP
from torchmetrics.wrappers import BinaryTargetTransformer, LambdaInputTransformer, MetricInputTransformer
from unittests._helpers import seed_all

seed_all(42)


class TestMetricInputTransformer:
    """Test suite for MetricInputTransformer."""

    def test_no_base_metric(self) -> None:
        """Tests that TypeError is raised when no wrapped_metric is passed."""
        with pytest.raises(TypeError, match=r"Expected wrapped metric to be an instance of .*"):
            MetricInputTransformer([])


class TestLambdaInputTransformer:
    """Test suite for LambdaInputTransformer."""

    _test_signature: ClassVar = (
        "cls",
        "transform_pred",
        "transform_target",
        "preds",
        "preds_transformed",
        "targets",
        "targets_transformed",
    )
    _test_cases: ClassVar = [
        # No change to input data (identity transform)
        (
            BinaryAccuracy,
            None,
            None,
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.5, 0.4],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.5, 0.4],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ),
        # Change to pred data (invert transform)
        (
            BinaryAccuracy,
            lambda pred: 1 - pred,
            None,
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.5, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.5, 0.6],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ),
        # Change to target data (invert transform)
        (
            BinaryAccuracy,
            None,
            lambda target: 1 - target,
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.5, 0.4],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.5, 0.4],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ),
        # Change to all input data (invert transform)
        (
            BinaryAccuracy,
            lambda pred: 1 - pred,
            lambda target: 1 - target,
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.5, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.5, 0.6],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ),
    ]

    @pytest.mark.parametrize(_test_signature, _test_cases)
    def test_forward(
        self, cls, transform_pred, transform_target, preds, preds_transformed, targets, targets_transformed
    ) -> None:
        """Tests if the binarized forward matches the output of the metric on manually binarized targets."""
        metric = cls()
        wrapped_metric = LambdaInputTransformer(
            metric, transform_pred=transform_pred, transform_target=transform_target
        )
        preds = Tensor(preds).float()
        targets = Tensor(targets).float()
        preds_transformed = Tensor(preds_transformed).float()
        targets_transformed = Tensor(targets_transformed).float() if targets_transformed is not None else targets

        args = (preds, targets)
        transformed_args = (preds_transformed, targets_transformed)
        assert metric(*transformed_args) == wrapped_metric(*args)

    @pytest.mark.parametrize(_test_signature, _test_cases)
    def test_update(
        self, cls, transform_pred, transform_target, preds, preds_transformed, targets, targets_transformed
    ) -> None:
        """Tests if the binarized update matches the output of the metric on manually binarized targets."""
        metric = cls()
        wrapped_metric = LambdaInputTransformer(
            metric, transform_pred=transform_pred, transform_target=transform_target
        )
        preds = Tensor(preds).float()
        targets = Tensor(targets).float()
        preds_transformed = Tensor(preds_transformed).float() if preds_transformed is not None else preds
        targets_transformed = Tensor(targets_transformed).float() if targets_transformed is not None else targets

        args = (preds, targets)
        transformed_args = (preds_transformed, targets_transformed)
        metric.update(*transformed_args)
        wrapped_metric.update(*args)
        assert metric.compute() == wrapped_metric.compute()

    def test_no_transform_pred(self) -> None:
        """Tests that TypeError is raised when a non-callable is passed as `transform_pred`."""
        with pytest.raises(TypeError, match=r"Expected `transform_pred` to be of type .*"):
            LambdaInputTransformer(BinaryAccuracy(), transform_pred=[])

    def test_no_transform_target(self) -> None:
        """Tests that TypeError is raised when a non-callable is passed as `transform_target`."""
        with pytest.raises(TypeError, match=r"Expected `transform_target` to be of type .*"):
            LambdaInputTransformer(BinaryAccuracy(), transform_target=[])


class TestBinaryTargetTransformer:
    """Test class for BinaryTargetTransformer."""

    _test_signature: ClassVar = ("cls", "threshold", "preds", "targets", "targets_binary", "kwargs")
    _test_cases: ClassVar = [
        # Metric with targets and with kwargs
        (
            RetrievalMAP,
            0,
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.5, 0.4],
            [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            {"indexes": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
        ),
        (
            RetrievalMAP,
            1,
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.5, 0.4],
            [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            {"indexes": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
        ),
        (
            RetrievalMAP,
            2,
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.6, 0.7, 0.8, 0.5, 0.4],
            [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            {"indexes": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
        ),
        # Metric with targets and without kwargs
        (
            BinaryAccuracy,
            0,
            [0.9, 0.8, 0.9, 0.6, 0.5, 0.4, 0.4, 0.1, 0.5, 0.4],
            [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            {},
        ),
        (
            BinaryAccuracy,
            1,
            [0.9, 0.8, 0.9, 0.6, 0.5, 0.4, 0.4, 0.1, 0.5, 0.4],
            [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            {},
        ),
        (
            BinaryAccuracy,
            2,
            [0.9, 0.8, 0.9, 0.6, 0.5, 0.4, 0.4, 0.1, 0.5, 0.4],
            [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            {},
        ),
        # Metric without targets and without kwargs
        (MeanMetric, 2, [0.9, 0.8, 0.9, 0.6, 0.5, 0.4, 0.4, 0.1, 0.5, 0.4], None, None, {}),
    ]

    @pytest.mark.parametrize(_test_signature, _test_cases)
    def test_forward(self, cls, threshold, preds, targets, targets_binary, kwargs) -> None:
        """Tests if the binarized forward matches the output of the metric on manually binarized targets."""
        metric = cls()
        wrapped_metric = BinaryTargetTransformer(metric, threshold=threshold)
        preds = Tensor(preds).float()
        targets = Tensor(targets).float() if targets is not None else None
        targets_binary = Tensor(targets_binary).float() if targets_binary is not None else None

        args = (preds, targets) if targets is not None else (preds,)
        wrapped_args = (preds, targets_binary) if targets_binary is not None else (preds,)
        kwargs = {k: Tensor(v).long() for k, v in kwargs.items()}
        assert metric(*wrapped_args, **kwargs) == wrapped_metric(*args, **kwargs)

    @pytest.mark.parametrize(_test_signature, _test_cases)
    def test_update(self, cls, threshold, preds, targets, targets_binary, kwargs) -> None:
        """Tests if the binarized update matches the output of the metric on manually binarized targets."""
        metric = cls()
        wrapped_metric = BinaryTargetTransformer(metric, threshold=threshold)
        preds = Tensor(preds).float()
        if targets is not None:
            targets = Tensor(targets).float()
            targets_binary = Tensor(targets_binary).float()

        args = (preds, targets) if targets is not None else [preds]
        wrapped_args = (preds, targets_binary) if targets_binary is not None else [preds]
        kwargs = {k: Tensor(v).long() for k, v in kwargs.items()}
        metric.update(*wrapped_args, **kwargs)
        wrapped_metric.update(*args, **kwargs)
        assert metric.compute() == wrapped_metric.compute()

    def test_threshold(self) -> None:
        """Tests that TypeError is raised when invalid threshold is passed."""
        with pytest.raises(TypeError, match=r"Expected `threshold` to be of type .*"):
            BinaryTargetTransformer(RetrievalMAP(), "a")

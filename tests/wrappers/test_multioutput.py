from collections import namedtuple
from functools import partial
from typing import Any, Callable, Optional

import pytest
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score as sk_r2score

from tests.helpers import seed_all
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, MetricTester
from torchmetrics import Metric
from torchmetrics.classification import Accuracy
from torchmetrics.regression import R2Score
from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_6
from torchmetrics.wrappers.multioutput import MultioutputWrapper

seed_all(42)


class _MultioutputMetric(Metric):
    """Multi-output version of the R2 score class for testing."""

    def __init__(
        self,
        base_metric_class,
        num_outputs: int = 1,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Any = None,
        dist_sync_fn: Optional[Callable] = None,
        **base_metric_kwargs,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.metric = MultioutputWrapper(
            base_metric_class(**base_metric_kwargs),
            num_outputs=num_outputs,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            dist_sync_fn=dist_sync_fn,
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the each pair of outputs and predictions."""
        return self.metric.update(preds, target)

    def compute(self) -> torch.Tensor:
        """Compute the R2 score between each pair of outputs and predictions."""
        return self.metric.compute()

    @torch.jit.unused
    def forward(self, *args, **kwargs):
        return self.metric(*args, **kwargs)

    def reset(self) -> None:
        """Reset the underlying metric state."""
        self.metric.reset()


num_classes = 3
num_targets = 2

Input = namedtuple("Input", ["preds", "target"])

_multi_target_regression_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets), target=torch.rand(NUM_BATCHES, BATCH_SIZE, num_targets),
)
_multi_target_classification_inputs = Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, num_classes, num_targets),
    target=torch.randint(num_classes, (NUM_BATCHES, BATCH_SIZE, num_targets)),
)


def _multi_target_sk_r2score(preds, target, adjusted=0, multioutput="raw_values"):
    """Compute R2 score over multiple outputs."""
    sk_preds = preds.view(-1, num_targets).numpy()
    sk_target = target.view(-1, num_targets).numpy()
    r2_score = sk_r2score(sk_target, sk_preds, multioutput=multioutput)
    if adjusted != 0:
        r2_score = 1 - (1 - r2_score) * (sk_preds.shape[0] - 1) / (sk_preds.shape[0] - adjusted - 1)
    return r2_score


<<<<<<< HEAD
=======
def _multi_target_sk_accuracy(preds, target, num_outputs):
    """Compute accuracy over multiple outputs."""
    accs = []
    for i in range(num_outputs):
        accs.append(accuracy_score(torch.argmax(preds[:, :, i], dim=1), target[:, i]))
    return accs


>>>>>>> Fix DeepSource flagged issues
@pytest.mark.parametrize(
    "base_metric_class, compare_metric, preds, target, num_outputs, metric_kwargs",
    [
        (
            R2Score,
            _multi_target_sk_r2score,
            _multi_target_regression_inputs.preds,
            _multi_target_regression_inputs.target,
            num_targets,
            {},
        ),
        (
            Accuracy,
            partial(_multi_target_sk_accuracy, num_outputs=2),
            _multi_target_classification_inputs.preds,
            _multi_target_classification_inputs.target,
            num_targets,
            dict(num_classes=num_classes),
        ),
    ],
)
class TestMultioutputWrapper(MetricTester):
    """Test the MultioutputWrapper class with regression and classification inner metrics."""
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_multioutput_wrapper(
        self, base_metric_class, compare_metric, preds, target, num_outputs, metric_kwargs, ddp, dist_sync_on_step
    ):
        """Test that the multioutput wrapper properly slices and computes outputs along the output dimension for
        both classification and regression metrics."""
        self.run_class_metric_test(
            ddp,
            preds,
            target,
            _MultioutputMetric,
            compare_metric,
            dist_sync_on_step,
            metric_args=dict(num_outputs=num_outputs, base_metric_class=base_metric_class, **metric_kwargs),
        )

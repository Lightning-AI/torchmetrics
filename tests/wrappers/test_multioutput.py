from functools import partial

import pytest
import torch
from sklearn.metrics import accuracy_score

from torchmetrics.classification import Accuracy
from torchmetrics.regression import R2Score
from torchmetrics.wrappers.multioutput import MultioutputWrapper


def _multioutput_sk_accuracy(preds, target, num_outputs):
    accs = []
    for i in range(num_outputs):
        accs.append(accuracy_score(torch.argmax(preds[:, :, i], dim=1), target[:, i]))
    return accs


@pytest.mark.parametrize(
    "metric, compare_metric, pred_generator, target_generator, num_rounds",
    [
        (
            MultioutputWrapper(R2Score(), num_outputs=2),
            R2Score(num_outputs=2, multioutput="raw_values"),
            partial(torch.randn, 10, 2),
            partial(torch.randn, 10, 2),
            2,
        ),
        (
            MultioutputWrapper(Accuracy(num_classes=3), num_outputs=2),
            partial(_multioutput_sk_accuracy, num_outputs=2),
            partial(torch.rand, 10, 3, 2),
            partial(torch.randint, 3, (10, 2)),
            2,
        ),
    ],
)
def test_multioutput_wrapper(metric, compare_metric, pred_generator, target_generator, num_rounds):
    """Test that the multioutput wrapper properly slices and computes outputs along the output dimension for both
    classification and regression metrics."""
    preds, targets = [], []
    for _ in range(num_rounds):
        preds.append(pred_generator())
        targets.append(target_generator())
        print(preds[-1].shape, targets[-1].shape)
        metric.update(preds[-1], targets[-1])
    expected_metric_val = compare_metric(torch.cat(preds), torch.cat(targets))
    torch.testing.assert_allclose(metric.compute(), expected_metric_val)

# TODO: replace this with official implementation once PR is merged!
from ._sklearn_calibration import calibration_error as sk_calib
import numpy as np
import pytest
from torch import tensor
from tests.classification.inputs import _input_multiclass as _input_mcls
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester
from torchmetrics import ExpectedCalibrationError, metric
from torchmetrics.functional import expected_calibration_error
from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.utilities.enums import DataType
from tests.classification.inputs import _input_binary, _input_binary_prob
from tests.classification.inputs import _input_multiclass as _input_mcls
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass as _input_mdmc
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
import functools

seed_all(42)


def _sk_ece(preds, target, n_bins):
    _, _, mode = _input_format_classification(preds, target, threshold=THRESHOLD)
    sk_preds, sk_target = preds.numpy(), target.numpy()

    if mode == DataType.MULTICLASS:
        # binary label is whether or not the predicted class is correct
        sk_target = np.equal(np.argmax(sk_preds, axis=1), sk_target)
        sk_preds = np.max(sk_preds, axis=1)

    return sk_calib(y_true=sk_target, y_prob=sk_preds, norm="l1", n_bins=n_bins)


def _sk_mce(preds, target, n_bins):
    return 0


@ pytest.mark.parametrize(
    "preds, target",
    [(_input_binary_prob.preds, _input_binary_prob.target),
     (_input_mcls_prob.preds, _input_mcls_prob.target),
     ]
)
class TestECE(MetricTester):

    @pytest.mark.parametrize("n_bins", [10, 15, 20])
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_ece(self, preds, target, n_bins, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=ExpectedCalibrationError,
            sk_metric=functools.partial(_sk_ece, n_bins=n_bins),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"n_bins": n_bins})

    @ pytest.mark.parametrize("n_bins", [10, 15, 20])
    def test_ece_functional(self, preds, target, n_bins):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=expected_calibration_error,
            sk_metric=functools.partial(_sk_ece, n_bins=n_bins),
            metric_args={"n_bins": n_bins})


@pytest.mark.parametrize("preds, targets", [(_input_mdmc_prob.preds, _input_mdmc_prob.target), (_input_mlb_prob.preds, _input_mlb_prob.target)])
def test_invalid_input(preds, targets):
    with pytest.raises(ValueError):
        expected_calibration_error(preds, targets)

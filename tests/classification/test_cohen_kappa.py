from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import cohen_kappa_score as sk_cohen_kappa

from tests.classification.inputs import _input_binary, _input_binary_prob
from tests.classification.inputs import _input_multiclass as _input_mcls
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass as _input_mdmc
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester
from torchmetrics.classification.cohen_kappa import CohenKappa
from torchmetrics.functional.classification.cohen_kappa import cohen_kappa

seed_all(42)


def _sk_cohen_kappa_binary_prob(preds, target, weights=None):
    sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


def _sk_cohen_kappa_binary(preds, target, weights=None):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


def _sk_cohen_kappa_multilabel_prob(preds, target, weights=None):
    sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


def _sk_cohen_kappa_multilabel(preds, target, weights=None):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


def _sk_cohen_kappa_multiclass_prob(preds, target, weights=None):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 1).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


def _sk_cohen_kappa_multiclass(preds, target, weights=None):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


def _sk_cohen_kappa_multidim_multiclass_prob(preds, target, weights=None):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 2).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


def _sk_cohen_kappa_multidim_multiclass(preds, target, weights=None):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_cohen_kappa(y1=sk_target, y2=sk_preds, weights=weights)


@pytest.mark.parametrize("weights", ["linear", "quadratic", None])
@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes",
    [
        (_input_binary_prob.preds, _input_binary_prob.target, _sk_cohen_kappa_binary_prob, 2),
        (_input_binary.preds, _input_binary.target, _sk_cohen_kappa_binary, 2),
        (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_cohen_kappa_multilabel_prob, 2),
        (_input_mlb.preds, _input_mlb.target, _sk_cohen_kappa_multilabel, 2),
        (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_cohen_kappa_multiclass_prob, NUM_CLASSES),
        (_input_mcls.preds, _input_mcls.target, _sk_cohen_kappa_multiclass, NUM_CLASSES),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_cohen_kappa_multidim_multiclass_prob, NUM_CLASSES),
        (_input_mdmc.preds, _input_mdmc.target, _sk_cohen_kappa_multidim_multiclass, NUM_CLASSES),
    ],
)
class TestCohenKappa(MetricTester):
    atol = 1e-5

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_cohen_kappa(self, weights, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=CohenKappa,
            sk_metric=partial(sk_metric, weights=weights),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "weights": weights},
        )

    def test_cohen_kappa_functional(self, weights, preds, target, sk_metric, num_classes):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=cohen_kappa,
            sk_metric=partial(sk_metric, weights=weights),
            metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "weights": weights},
        )

    def test_cohen_kappa_differentiability(self, preds, target, sk_metric, weights, num_classes):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=CohenKappa,
            metric_functional=cohen_kappa,
            metric_args={"num_classes": num_classes, "threshold": THRESHOLD, "weights": weights},
        )


def test_warning_on_wrong_weights(tmpdir):
    preds = torch.randint(3, size=(20,))
    target = torch.randint(3, size=(20,))

    with pytest.raises(ValueError, match=".* ``weights`` but should be either None, 'linear' or 'quadratic'"):
        cohen_kappa(preds, target, num_classes=3, weights="unknown_arg")

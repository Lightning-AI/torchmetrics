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
from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_curve as sk_roc_curve
from torch import tensor

from tests.classification.inputs import _input_binary_prob
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel_multidim_prob as _input_mlmd_prob
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, MetricTester
from torchmetrics.classification.roc import ROC
from torchmetrics.functional import roc

seed_all(42)


def _sk_roc_curve(y_true, probas_pred, num_classes: int = 1, multilabel: bool = False):
    """ Adjusted comparison function that can also handles multiclass """
    if num_classes == 1:
        return sk_roc_curve(y_true, probas_pred, drop_intermediate=False)

    fpr, tpr, thresholds = [], [], []
    for i in range(num_classes):
        if multilabel:
            y_true_temp = y_true[:, i]
        else:
            y_true_temp = np.zeros_like(y_true)
            y_true_temp[y_true == i] = 1

        res = sk_roc_curve(y_true_temp, probas_pred[:, i], drop_intermediate=False)
        fpr.append(res[0])
        tpr.append(res[1])
        thresholds.append(res[2])
    return fpr, tpr, thresholds


def _sk_roc_binary_prob(preds, target, num_classes=1):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return _sk_roc_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


def _sk_roc_multiclass_prob(preds, target, num_classes=1):
    sk_preds = preds.reshape(-1, num_classes).numpy()
    sk_target = target.view(-1).numpy()

    return _sk_roc_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


def _sk_roc_multidim_multiclass_prob(preds, target, num_classes=1):
    sk_preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
    sk_target = target.view(-1).numpy()
    return _sk_roc_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes)


def _sk_roc_multilabel_prob(preds, target, num_classes=1):
    sk_preds = preds.numpy()
    sk_target = target.numpy()
    return _sk_roc_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes, multilabel=True)


def _sk_roc_multilabel_multidim_prob(preds, target, num_classes=1):
    sk_preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
    sk_target = target.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
    return _sk_roc_curve(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes, multilabel=True)


@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes",
    [(_input_binary_prob.preds, _input_binary_prob.target, _sk_roc_binary_prob, 1),
     (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_roc_multiclass_prob, NUM_CLASSES),
     (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_roc_multidim_multiclass_prob, NUM_CLASSES),
     (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_roc_multilabel_prob, NUM_CLASSES),
     (_input_mlmd_prob.preds, _input_mlmd_prob.target, _sk_roc_multilabel_multidim_prob, NUM_CLASSES)]
)
class TestROC(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_roc(self, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=ROC,
            sk_metric=partial(sk_metric, num_classes=num_classes),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"num_classes": num_classes}
        )

    def test_roc_functional(self, preds, target, sk_metric, num_classes):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=roc,
            sk_metric=partial(sk_metric, num_classes=num_classes),
            metric_args={"num_classes": num_classes},
        )

    def test_roc_differentiability(self, preds, target, sk_metric, num_classes):
        self.run_differentiability_test(
            preds,
            target,
            metric_module=ROC,
            metric_functional=roc,
            metric_args={"num_classes": num_classes},
        )


@pytest.mark.parametrize(['pred', 'target', 'expected_tpr', 'expected_fpr'], [
    pytest.param([0, 1], [0, 1], [0, 1, 1], [0, 0, 1]),
    pytest.param([1, 0], [0, 1], [0, 0, 1], [0, 1, 1]),
    pytest.param([1, 1], [1, 0], [0, 1], [0, 1]),
    pytest.param([1, 0], [1, 0], [0, 1, 1], [0, 0, 1]),
    pytest.param([0.5, 0.5], [0, 1], [0, 1], [0, 1]),
])
def test_roc_curve(pred, target, expected_tpr, expected_fpr):
    fpr, tpr, thresh = roc(tensor(pred), tensor(target))

    assert fpr.shape == tpr.shape
    assert fpr.size(0) == thresh.size(0)
    assert torch.allclose(fpr, tensor(expected_fpr).to(fpr))
    assert torch.allclose(tpr, tensor(expected_tpr).to(tpr))

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
import numpy as np
import pytest
import torch
from sklearn.metrics import matthews_corrcoef as sk_matthews_corrcoef

from tests.classification.inputs import _input_binary, _input_binary_prob
from tests.classification.inputs import _input_multiclass as _input_mcls
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass as _input_mdmc
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrcoef
from torchmetrics.functional.classification.matthews_corrcoef import matthews_corrcoef

seed_all(42)


def _sk_matthews_corrcoef_binary_prob(preds, target):
    sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


def _sk_matthews_corrcoef_binary(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


def _sk_matthews_corrcoef_multilabel_prob(preds, target):
    sk_preds = (preds.view(-1).numpy() >= THRESHOLD).astype(np.uint8)
    sk_target = target.view(-1).numpy()

    return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


def _sk_matthews_corrcoef_multilabel(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


def _sk_matthews_corrcoef_multiclass_prob(preds, target):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 1).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


def _sk_matthews_corrcoef_multiclass(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


def _sk_matthews_corrcoef_multidim_multiclass_prob(preds, target):
    sk_preds = torch.argmax(preds, dim=len(preds.shape) - 2).view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


def _sk_matthews_corrcoef_multidim_multiclass(preds, target):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return sk_matthews_corrcoef(y_true=sk_target, y_pred=sk_preds)


@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes",
    [(_input_binary_prob.preds, _input_binary_prob.target, _sk_matthews_corrcoef_binary_prob, 2),
     (_input_binary.preds, _input_binary.target, _sk_matthews_corrcoef_binary, 2),
     (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_matthews_corrcoef_multilabel_prob, 2),
     (_input_mlb.preds, _input_mlb.target, _sk_matthews_corrcoef_multilabel, 2),
     (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_matthews_corrcoef_multiclass_prob, NUM_CLASSES),
     (_input_mcls.preds, _input_mcls.target, _sk_matthews_corrcoef_multiclass, NUM_CLASSES),
     (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_matthews_corrcoef_multidim_multiclass_prob, NUM_CLASSES),
     (_input_mdmc.preds, _input_mdmc.target, _sk_matthews_corrcoef_multidim_multiclass, NUM_CLASSES)]
)
class TestMatthewsCorrCoef(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_matthews_corrcoef(self, preds, target, sk_metric, num_classes, ddp, dist_sync_on_step):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=MatthewsCorrcoef,
            sk_metric=sk_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "threshold": THRESHOLD,
            }
        )

    def test_matthews_corrcoef_functional(self, preds, target, sk_metric, num_classes):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=matthews_corrcoef,
            sk_metric=sk_metric,
            metric_args={
                "num_classes": num_classes,
                "threshold": THRESHOLD,
            }
        )

    def test_matthews_corrcoef_differentiability(self, preds, target, sk_metric, num_classes):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=MatthewsCorrcoef,
            metric_functional=matthews_corrcoef,
            metric_args={
                "num_classes": num_classes,
                "threshold": THRESHOLD,
            }
        )

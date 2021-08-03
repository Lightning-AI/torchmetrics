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

import pytest
import torch
from sklearn.metrics import roc_auc_score as sk_roc_auc_score

from tests.classification.inputs import _input_binary_prob
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel_multidim_prob as _input_mlmd_prob
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, MetricTester
from torchmetrics.classification.auroc import AUROC
from torchmetrics.functional import auroc
from torchmetrics.utilities.imports import _TORCH_LOWER_1_6

seed_all(42)


def _sk_auroc_binary_prob(preds, target, num_classes, average="macro", max_fpr=None, multi_class="ovr"):
    # todo: `multi_class` is unused
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()
    return sk_roc_auc_score(y_true=sk_target, y_score=sk_preds, average=average, max_fpr=max_fpr)


def _sk_auroc_multiclass_prob(preds, target, num_classes, average="macro", max_fpr=None, multi_class="ovr"):
    sk_preds = preds.reshape(-1, num_classes).numpy()
    sk_target = target.view(-1).numpy()
    return sk_roc_auc_score(
        y_true=sk_target,
        y_score=sk_preds,
        average=average,
        max_fpr=max_fpr,
        multi_class=multi_class,
    )


def _sk_auroc_multidim_multiclass_prob(preds, target, num_classes, average="macro", max_fpr=None, multi_class="ovr"):
    sk_preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
    sk_target = target.view(-1).numpy()
    return sk_roc_auc_score(
        y_true=sk_target,
        y_score=sk_preds,
        average=average,
        max_fpr=max_fpr,
        multi_class=multi_class,
    )


def _sk_auroc_multilabel_prob(preds, target, num_classes, average="macro", max_fpr=None, multi_class="ovr"):
    sk_preds = preds.reshape(-1, num_classes).numpy()
    sk_target = target.reshape(-1, num_classes).numpy()
    return sk_roc_auc_score(
        y_true=sk_target,
        y_score=sk_preds,
        average=average,
        max_fpr=max_fpr,
        multi_class=multi_class,
    )


def _sk_auroc_multilabel_multidim_prob(preds, target, num_classes, average="macro", max_fpr=None, multi_class="ovr"):
    sk_preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
    sk_target = target.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
    return sk_roc_auc_score(
        y_true=sk_target,
        y_score=sk_preds,
        average=average,
        max_fpr=max_fpr,
        multi_class=multi_class,
    )


@pytest.mark.parametrize("average", ["macro", "weighted", "micro"])
@pytest.mark.parametrize("max_fpr", [None, 0.8, 0.5])
@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes",
    [
        (_input_binary_prob.preds, _input_binary_prob.target, _sk_auroc_binary_prob, 1),
        (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_auroc_multiclass_prob, NUM_CLASSES),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_auroc_multidim_multiclass_prob, NUM_CLASSES),
        (_input_mlb_prob.preds, _input_mlb_prob.target, _sk_auroc_multilabel_prob, NUM_CLASSES),
        (_input_mlmd_prob.preds, _input_mlmd_prob.target, _sk_auroc_multilabel_multidim_prob, NUM_CLASSES),
    ],
)
class TestAUROC(MetricTester):
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_auroc(self, preds, target, sk_metric, num_classes, average, max_fpr, ddp, dist_sync_on_step):
        # max_fpr different from None is not support in multi class
        if max_fpr is not None and num_classes != 1:
            pytest.skip("max_fpr parameter not support for multi class or multi label")

        # max_fpr only supported for torch v1.6 or higher
        if max_fpr is not None and _TORCH_LOWER_1_6:
            pytest.skip("requires torch v1.6 or higher to test max_fpr argument")

        # average='micro' only supported for multilabel
        if average == "micro" and preds.ndim > 2 and preds.ndim == target.ndim + 1:
            pytest.skip("micro argument only support for multilabel input")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=AUROC,
            sk_metric=partial(sk_metric, num_classes=num_classes, average=average, max_fpr=max_fpr),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"num_classes": num_classes, "average": average, "max_fpr": max_fpr},
        )

    def test_auroc_functional(self, preds, target, sk_metric, num_classes, average, max_fpr):
        # max_fpr different from None is not support in multi class
        if max_fpr is not None and num_classes != 1:
            pytest.skip("max_fpr parameter not support for multi class or multi label")

        # max_fpr only supported for torch v1.6 or higher
        if max_fpr is not None and _TORCH_LOWER_1_6:
            pytest.skip("requires torch v1.6 or higher to test max_fpr argument")

        # average='micro' only supported for multilabel
        if average == "micro" and preds.ndim > 2 and preds.ndim == target.ndim + 1:
            pytest.skip("micro argument only support for multilabel input")

        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=auroc,
            sk_metric=partial(sk_metric, num_classes=num_classes, average=average, max_fpr=max_fpr),
            metric_args={"num_classes": num_classes, "average": average, "max_fpr": max_fpr},
        )

    def test_auroc_differentiability(self, preds, target, sk_metric, num_classes, average, max_fpr):
        # max_fpr different from None is not support in multi class
        if max_fpr is not None and num_classes != 1:
            pytest.skip("max_fpr parameter not support for multi class or multi label")

        # max_fpr only supported for torch v1.6 or higher
        if max_fpr is not None and _TORCH_LOWER_1_6:
            pytest.skip("requires torch v1.6 or higher to test max_fpr argument")

        # average='micro' only supported for multilabel
        if average == "micro" and preds.ndim > 2 and preds.ndim == target.ndim + 1:
            pytest.skip("micro argument only support for multilabel input")

        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=AUROC,
            metric_functional=auroc,
            metric_args={"num_classes": num_classes, "average": average, "max_fpr": max_fpr},
        )


def test_error_on_different_mode():
    """test that an error is raised if the user pass in data of
    different modes (binary, multi-label, multi-class)
    """
    metric = AUROC()
    # pass in multi-class data
    metric.update(torch.randn(10, 5).softmax(dim=-1), torch.randint(0, 5, (10,)))
    with pytest.raises(ValueError, match=r"The mode of data.* should be constant.*"):
        # pass in multi-label data
        metric.update(torch.rand(10, 5), torch.randint(0, 2, (10, 5)))


def test_error_multiclass_no_num_classes():
    with pytest.raises(
        ValueError, match="Detected input to `multiclass` but you did not provide `num_classes` argument"
    ):
        _ = auroc(torch.randn(20, 3).softmax(dim=-1), torch.randint(3, (20,)))


def test_weighted_with_empty_classes():
    """Tests that weighted multiclass AUROC calculation yields the same results if a new
    but empty class exists. Tests that the proper warnings and errors are raised
    """
    preds = torch.tensor(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
            [0.85, 0.05, 0.10],
            [0.10, 0.10, 0.80],
        ]
    )
    target = torch.tensor([0, 1, 1, 2, 2])
    num_classes = 3
    _auroc = auroc(preds, target, average="weighted", num_classes=num_classes)

    # Add in a class with zero observations at second to last index
    preds = torch.cat(
        (preds[:, : num_classes - 1], torch.rand_like(preds[:, 0:1]), preds[:, num_classes - 1 :]), axis=1
    )
    # Last class (2) gets moved to 3
    target[target == num_classes - 1] = num_classes
    with pytest.warns(UserWarning, match="Class 2 had 0 observations, omitted from AUROC calculation"):
        _auroc_empty_class = auroc(preds, target, average="weighted", num_classes=num_classes + 1)
    assert _auroc == _auroc_empty_class

    target = torch.zeros_like(target)
    with pytest.raises(ValueError, match="Found 1 non-empty class in `multiclass` AUROC calculation"):
        _ = auroc(preds, target, average="weighted", num_classes=num_classes + 1)

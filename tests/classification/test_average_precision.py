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
from sklearn.metrics import average_precision_score as sk_average_precision_score
from torch import tensor

from tests.classification.inputs import _input_binary_prob
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, MetricTester
from torchmetrics.classification.avg_precision import AveragePrecision
from torchmetrics.functional import average_precision

seed_all(42)


def _sk_average_precision_score(y_true, probas_pred, num_classes=1, average=None):
    if num_classes == 1:
        return sk_average_precision_score(y_true, probas_pred)

    res = []
    for i in range(num_classes):
        y_true_temp = np.zeros_like(y_true)
        y_true_temp[y_true == i] = 1
        res.append(sk_average_precision_score(y_true_temp, probas_pred[:, i]))

    if average == "macro":
        return np.array(res).mean()
    if average == "weighted":
        weights = np.bincount(y_true) if y_true.max() > 1 else y_true.sum(axis=0)
        weights = weights / sum(weights)
        return (np.array(res) * weights).sum()

    return res


def _sk_avg_prec_binary_prob(preds, target, num_classes=1, average=None):
    sk_preds = preds.view(-1).numpy()
    sk_target = target.view(-1).numpy()

    return _sk_average_precision_score(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes, average=average)


def _sk_avg_prec_multiclass_prob(preds, target, num_classes=1, average=None):
    sk_preds = preds.reshape(-1, num_classes).numpy()
    sk_target = target.view(-1).numpy()

    return _sk_average_precision_score(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes, average=average)


def _sk_avg_prec_multilabel_prob(preds, target, num_classes=1, average=None):
    sk_preds = preds.reshape(-1, num_classes).numpy()
    sk_target = target.view(-1, num_classes).numpy()
    return sk_average_precision_score(sk_target, sk_preds, average=average)


def _sk_avg_prec_multidim_multiclass_prob(preds, target, num_classes=1, average=None):
    sk_preds = preds.transpose(0, 1).reshape(num_classes, -1).transpose(0, 1).numpy()
    sk_target = target.view(-1).numpy()
    return _sk_average_precision_score(y_true=sk_target, probas_pred=sk_preds, num_classes=num_classes, average=average)


@pytest.mark.parametrize(
    "preds, target, sk_metric, num_classes",
    [
        (_input_binary_prob.preds, _input_binary_prob.target, _sk_avg_prec_binary_prob, 1),
        (_input_mcls_prob.preds, _input_mcls_prob.target, _sk_avg_prec_multiclass_prob, NUM_CLASSES),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, _sk_avg_prec_multidim_multiclass_prob, NUM_CLASSES),
        (_input_multilabel.preds, _input_multilabel.target, _sk_avg_prec_multilabel_prob, NUM_CLASSES),
    ],
)
class TestAveragePrecision(MetricTester):
    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_average_precision(self, preds, target, sk_metric, num_classes, average, ddp, dist_sync_on_step):
        if target.max() > 1 and average == "micro":
            pytest.skip("average=micro and multiclass input cannot be used together")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=AveragePrecision,
            sk_metric=partial(sk_metric, num_classes=num_classes, average=average),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={"num_classes": num_classes, "average": average},
        )

    @pytest.mark.parametrize("average", ["micro", "macro", "weighted", None])
    def test_average_precision_functional(self, preds, target, sk_metric, num_classes, average):
        if target.max() > 1 and average == "micro":
            pytest.skip("average=micro and multiclass input cannot be used together")

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=average_precision,
            sk_metric=partial(sk_metric, num_classes=num_classes, average=average),
            metric_args={"num_classes": num_classes, "average": average},
        )

    def test_average_precision_differentiability(self, preds, sk_metric, target, num_classes):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=AveragePrecision,
            metric_functional=average_precision,
            metric_args={"num_classes": num_classes},
        )


@pytest.mark.parametrize(
    ["scores", "target", "expected_score"],
    [
        # Check the average_precision_score of a constant predictor is
        # the TPR
        # Generate a dataset with 25% of positives
        # And a constant score
        # The precision is then the fraction of positive whatever the recall
        # is, as there is only one threshold:
        (tensor([1, 1, 1, 1]), tensor([0, 0, 0, 1]), 0.25),
        # With threshold 0.8 : 1 TP and 2 TN and one FN
        (tensor([0.6, 0.7, 0.8, 9]), tensor([1, 0, 0, 1]), 0.75),
    ],
)
def test_average_precision(scores, target, expected_score):
    assert average_precision(scores, target) == expected_score


def test_average_precision_warnings_and_errors():
    """Test that the correct errors and warnings gets raised."""

    # check average argument
    with pytest.raises(ValueError, match="Expected argument `average` to be one .*"):
        AveragePrecision(num_classes=5, average="samples")

    # check that micro average cannot be used with multilabel input
    pred = tensor(
        [
            [0.75, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.75, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.75, 0.05, 0.05],
            [0.05, 0.05, 0.05, 0.75, 0.05],
        ]
    )
    target = tensor([0, 1, 3, 2])
    average_precision = AveragePrecision(num_classes=5, average="micro")
    with pytest.raises(ValueError, match="Cannot use `micro` average with multi-class input"):
        average_precision(pred, target)

    # check that warning is thrown when average=macro and nan is encoutered in individual scores
    average_precision = AveragePrecision(num_classes=5, average="macro")
    with pytest.warns(UserWarning, match="Average precision score for one or more classes was `nan`.*"):
        average_precision(pred, target)

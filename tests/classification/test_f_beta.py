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
from typing import Callable, Optional

import numpy as np
import pytest
import torch
from sklearn.metrics import fbeta_score, f1_score

from tests.classification.inputs import _input_binary, _input_binary_prob
from tests.classification.inputs import _input_multiclass as _input_mcls
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass as _input_mdmc
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester
from torchmetrics import Metric, FBeta, F1
from torchmetrics.functional import fbeta, f1
from torchmetrics.utilities.checks import _input_format_classification

torch.manual_seed(42)


def _sk_fbeta_f1(preds, target, sk_fn, num_classes, average, is_multiclass, ignore_index, mdmc_average=None):
    if average == "none":
        average = None
    if num_classes == 1:
        average = "binary"

    labels = list(range(num_classes))
    try:
        labels.remove(ignore_index)
    except ValueError:
        pass

    sk_preds, sk_target, _ = _input_format_classification(
        preds, target, THRESHOLD, num_classes=num_classes, is_multiclass=is_multiclass
    )
    sk_preds, sk_target = sk_preds.numpy(), sk_target.numpy()

    sk_scores = sk_fn(sk_target, sk_preds, average=average, zero_division=0, labels=labels)

    if len(labels) != num_classes and not average:
        sk_scores = np.insert(sk_scores, ignore_index, np.nan)

    return sk_scores


def _sk_fbeta_f1_multidim_multiclass(
    preds, target, sk_fn, num_classes, average, is_multiclass, ignore_index, mdmc_average
):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=THRESHOLD, num_classes=num_classes, is_multiclass=is_multiclass
    )

    if mdmc_average == "global":
        preds = torch.transpose(preds, 1, 2).reshape(-1, preds.shape[1])
        target = torch.transpose(target, 1, 2).reshape(-1, target.shape[1])

        return _sk_fbeta_f1(preds, target, sk_fn, num_classes, average, False, ignore_index)
    elif mdmc_average == "samplewise":
        scores = []

        for i in range(preds.shape[0]):
            pred_i = preds[i, ...].T
            target_i = target[i, ...].T
            scores_i = _sk_fbeta_f1(pred_i, target_i, sk_fn, num_classes, average, False, ignore_index)

            scores.append(np.expand_dims(scores_i, 0))

        return np.concatenate(scores).mean(axis=0)


@pytest.mark.parametrize("metric_class, metric_fn, sk_fn", [
    (partial(FBeta, beta=2.0), partial(fbeta, beta=2.0), partial(fbeta_score, beta=2.0)), 
    (F1, f1, f1_score)
])
@pytest.mark.parametrize("average", ["micro", "macro", None, "weighted", "samples"])
@pytest.mark.parametrize("ignore_index", [None, 0])
@pytest.mark.parametrize(
    "preds, target, num_classes, is_multiclass, mdmc_average, sk_wrapper",
    [
        (_input_binary_prob.preds, _input_binary_prob.target, 1, None, None, _sk_fbeta_f1),
        (_input_binary.preds, _input_binary.target, 1, False, None, _sk_fbeta_f1),
        (_input_mlb_prob.preds, _input_mlb_prob.target, NUM_CLASSES, None, None, _sk_fbeta_f1),
        (_input_mlb.preds, _input_mlb.target, NUM_CLASSES, False, None, _sk_fbeta_f1),
        (_input_mcls_prob.preds, _input_mcls_prob.target, NUM_CLASSES, None, None, _sk_fbeta_f1),
        (_input_mcls.preds, _input_mcls.target, NUM_CLASSES, None, None, _sk_fbeta_f1),
        (_input_mdmc.preds, _input_mdmc.target, NUM_CLASSES, None, "global", _sk_fbeta_f1_multidim_multiclass),
        (
            _input_mdmc_prob.preds, _input_mdmc_prob.target, NUM_CLASSES, None, "global",
            _sk_fbeta_f1_multidim_multiclass
        ),
        (_input_mdmc.preds, _input_mdmc.target, NUM_CLASSES, None, "samplewise", _sk_fbeta_f1_multidim_multiclass),
        (
            _input_mdmc_prob.preds, _input_mdmc_prob.target, NUM_CLASSES, None, "samplewise",
            _sk_fbeta_f1_multidim_multiclass
        ),
    ],
)
class TestFBeta(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_fbeta_f1(
        self,
        ddp: bool,
        dist_sync_on_step: bool,
        preds: torch.Tensor,
        target: torch.Tensor,
        sk_wrapper: Callable,
        metric_class: Metric,
        metric_fn: Callable,
        sk_fn: Callable,
        is_multiclass: Optional[bool],
        num_classes: Optional[int],
        average: str,
        mdmc_average: Optional[str],
        ignore_index: Optional[int],
    ):
        if num_classes == 1 and average != "micro":
            pytest.skip("Only test binary data for 'micro' avg (equivalent of 'binary' in sklearn)")

        if ignore_index is not None and preds.ndim == 2:
            pytest.skip("Skipping ignore_index test with binary inputs.")

        if average == "weighted" and ignore_index is not None and mdmc_average is not None:
            pytest.skip("Ignore special case where we are ignoring entire sample for 'weighted' average")

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=metric_class,
            sk_metric=partial(
                sk_wrapper,
                sk_fn=sk_fn,
                average=average,
                num_classes=num_classes,
                is_multiclass=is_multiclass,
                ignore_index=ignore_index,
                mdmc_average=mdmc_average,
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "is_multiclass": is_multiclass,
                "ignore_index": ignore_index,
                "mdmc_average": mdmc_average,
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )

    def test_fbeta_f1_functional(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sk_wrapper: Callable,
        metric_class: Metric,
        metric_fn: Callable,
        sk_fn: Callable,
        is_multiclass: Optional[bool],
        num_classes: Optional[int],
        average: str,
        mdmc_average: Optional[str],
        ignore_index: Optional[int],
    ):
        if num_classes == 1 and average != "micro":
            pytest.skip("Only test binary data for 'micro' avg (equivalent of 'binary' in sklearn)")

        if ignore_index is not None and preds.ndim == 2:
            pytest.skip("Skipping ignore_index test with binary inputs.")

        if average == "weighted" and ignore_index is not None and mdmc_average is not None:
            pytest.skip("Ignore special case where we are ignoring entire sample for 'weighted' average")

        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=metric_fn,
            sk_metric=partial(
                sk_wrapper,
                sk_fn=sk_fn,
                average=average,
                num_classes=num_classes,
                is_multiclass=is_multiclass,
                ignore_index=ignore_index,
                mdmc_average=mdmc_average,
            ),
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "is_multiclass": is_multiclass,
                "ignore_index": ignore_index,
                "mdmc_average": mdmc_average,
            },
        )

_mc_k_target = torch.tensor([0, 1, 2])
_mc_k_preds = torch.tensor([[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])
_ml_k_target = torch.tensor([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
_ml_k_preds = torch.tensor([[0.9, 0.2, 0.75], [0.1, 0.7, 0.8], [0.6, 0.1, 0.7]])

@pytest.mark.parametrize("metric_class, metric_fn", [
    (partial(FBeta, beta=2.0), partial(fbeta, beta=2.0)), 
    (F1, fbeta)
])
@pytest.mark.parametrize(
    "k, preds, target, average, expected_fbeta, expected_f1",
    [
        (1, _mc_k_preds, _mc_k_target, "micro", torch.tensor(2 / 3), torch.tensor(2 / 3)),
        (2, _mc_k_preds, _mc_k_target, "micro", torch.tensor(5 / 6), torch.tensor(2 / 3)),
        (1, _ml_k_preds, _ml_k_target, "micro", torch.tensor(0.0), torch.tensor(0.0)),
        (2, _ml_k_preds, _ml_k_target, "micro", torch.tensor(5 / 18), torch.tensor(2 / 9)),
    ],
)
def test_top_k(
    metric_class,
    metric_fn,
    k: int,
    preds: torch.Tensor,
    target: torch.Tensor,
    average: str,
    expected_fbeta: torch.Tensor,
    expected_f1: torch.Tensor,
):
    """A simple test to check that top_k works as expected.

    Just a sanity check, the tests in StatScores should already guarantee
    the corectness of results.
    """

    class_metric = metric_class(top_k=k, average=average, num_classes=3)
    class_metric.update(preds, target)

    if class_metric.beta != 1.0:
        result = expected_fbeta
    else:
        result = expected_f1
       
    assert torch.isclose(class_metric.compute(), result)
    assert torch.isclose(metric_fn(preds, target, top_k=k, average=average, num_classes=3), result)
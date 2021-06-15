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
import math
from functools import partial
from typing import Callable, Optional

import numpy as np
import pytest
import torch
from sklearn.metrics import multilabel_confusion_matrix
from torch import Tensor, tensor

from tests.classification.inputs import _input_binary, _input_binary_prob
from tests.classification.inputs import _input_multiclass as _input_mcls
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass as _input_mdmc
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, THRESHOLD, MetricTester
from torchmetrics import Metric, Specificity
from torchmetrics.functional import specificity
from torchmetrics.functional.classification.stat_scores import _reduce_stat_scores
from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.utilities.enums import AverageMethod

seed_all(42)


def _sk_stats_score(preds, target, reduce, num_classes, multiclass, ignore_index, top_k):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=THRESHOLD, num_classes=num_classes, multiclass=multiclass, top_k=top_k
    )
    sk_preds, sk_target = preds.numpy(), target.numpy()

    if reduce != "macro" and ignore_index is not None and preds.shape[1] > 1:
        sk_preds = np.delete(sk_preds, ignore_index, 1)
        sk_target = np.delete(sk_target, ignore_index, 1)

    if preds.shape[1] == 1 and reduce == "samples":
        sk_target = sk_target.T
        sk_preds = sk_preds.T

    sk_stats = multilabel_confusion_matrix(
        sk_target, sk_preds, samplewise=(reduce == "samples") and preds.shape[1] != 1
    )

    if preds.shape[1] == 1 and reduce != "samples":
        sk_stats = sk_stats[[1]].reshape(-1, 4)[:, [3, 1, 0, 2]]
    else:
        sk_stats = sk_stats.reshape(-1, 4)[:, [3, 1, 0, 2]]

    if reduce == "micro":
        sk_stats = sk_stats.sum(axis=0, keepdims=True)

    sk_stats = np.concatenate([sk_stats, sk_stats[:, [3]] + sk_stats[:, [0]]], 1)

    if reduce == "micro":
        sk_stats = sk_stats[0]

    if reduce == "macro" and ignore_index is not None and preds.shape[1]:
        sk_stats[ignore_index, :] = -1

    if reduce == "micro":
        _, fp, tn, _, _ = sk_stats
    else:
        _, fp, tn, _ = sk_stats[:, 0], sk_stats[:, 1], sk_stats[:, 2], sk_stats[:, 3]
    return fp, tn


def _sk_spec(preds, target, reduce, num_classes, multiclass, ignore_index, top_k=None, mdmc_reduce=None, stats=None):

    if stats:
        fp, tn = stats
    else:
        stats = _sk_stats_score(preds, target, reduce, num_classes, multiclass, ignore_index, top_k)
        fp, tn = stats

    fp, tn = tensor(fp), tensor(tn)
    spec = _reduce_stat_scores(
        numerator=tn,
        denominator=tn + fp,
        weights=None if reduce != "weighted" else tn + fp,
        average=reduce,
        mdmc_average=mdmc_reduce,
    )
    if reduce in [None, "none"] and ignore_index is not None and preds.shape[1] > 1:
        spec = spec.numpy()
        spec = np.insert(spec, ignore_index, math.nan)
        spec = tensor(spec)

    return spec


def _sk_spec_mdim_mcls(preds, target, reduce, mdmc_reduce, num_classes, multiclass, ignore_index, top_k=None):
    preds, target, _ = _input_format_classification(
        preds, target, threshold=THRESHOLD, num_classes=num_classes, multiclass=multiclass, top_k=top_k
    )

    if mdmc_reduce == "global":
        preds = torch.transpose(preds, 1, 2).reshape(-1, preds.shape[1])
        target = torch.transpose(target, 1, 2).reshape(-1, target.shape[1])
        return _sk_spec(preds, target, reduce, num_classes, False, ignore_index, top_k, mdmc_reduce)
    fp, tn = [], []
    stats = []

    for i in range(preds.shape[0]):
        pred_i = preds[i, ...].T
        target_i = target[i, ...].T
        fp_i, tn_i = _sk_stats_score(pred_i, target_i, reduce, num_classes, False, ignore_index, top_k)
        fp.append(fp_i)
        tn.append(tn_i)

    stats.append(fp)
    stats.append(tn)
    return _sk_spec(preds[0], target[0], reduce, num_classes, multiclass, ignore_index, top_k, mdmc_reduce, stats)


@pytest.mark.parametrize("metric, fn_metric", [(Specificity, specificity)])
@pytest.mark.parametrize(
    "average, mdmc_average, num_classes, ignore_index, match_str",
    [
        ("wrong", None, None, None, "`average`"),
        ("micro", "wrong", None, None, "`mdmc"),
        ("macro", None, None, None, "number of classes"),
        ("macro", None, 1, 0, "ignore_index"),
    ],
)
def test_wrong_params(metric, fn_metric, average, mdmc_average, num_classes, ignore_index, match_str):
    with pytest.raises(ValueError, match=match_str):
        metric(
            average=average,
            mdmc_average=mdmc_average,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    with pytest.raises(ValueError, match=match_str):
        fn_metric(
            _input_binary.preds[0],
            _input_binary.target[0],
            average=average,
            mdmc_average=mdmc_average,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )


@pytest.mark.parametrize("metric_class, metric_fn", [(Specificity, specificity)])
def test_zero_division(metric_class, metric_fn):
    """ Test that zero_division works correctly (currently should just set to 0). """

    preds = tensor([1, 2, 1, 1])
    target = tensor([0, 0, 0, 0])

    cl_metric = metric_class(average="none", num_classes=3)
    cl_metric(preds, target)

    result_cl = cl_metric.compute()
    result_fn = metric_fn(preds, target, average="none", num_classes=3)

    assert result_cl[0] == result_fn[0] == 0


@pytest.mark.parametrize("metric_class, metric_fn", [(Specificity, specificity)])
def test_no_support(metric_class, metric_fn):
    """This tests a rare edge case, where there is only one class present
    in target, and ignore_index is set to exactly that class - and the
    average method is equal to 'weighted'.

    This would mean that the sum of weights equals zero, and would, without
    taking care of this case, return NaN. However, the reduction function
    should catch that and set the metric to equal the value of zero_division
    in this case (zero_division is for now not configurable and equals 0).
    """

    preds = tensor([1, 1, 0, 0])
    target = tensor([0, 0, 0, 0])

    cl_metric = metric_class(average="weighted", num_classes=2, ignore_index=1)
    cl_metric(preds, target)

    result_cl = cl_metric.compute()
    result_fn = metric_fn(preds, target, average="weighted", num_classes=2, ignore_index=1)

    assert result_cl == result_fn == 0


@pytest.mark.parametrize("metric_class, metric_fn", [(Specificity, specificity)])
@pytest.mark.parametrize("average", ["micro", "macro", None, "weighted", "samples"])
@pytest.mark.parametrize("ignore_index", [None, 0])
@pytest.mark.parametrize(
    "preds, target, num_classes, multiclass, mdmc_average, sk_wrapper",
    [
        (_input_binary_prob.preds, _input_binary_prob.target, 1, None, None, _sk_spec),
        (_input_binary.preds, _input_binary.target, 1, False, None, _sk_spec),
        (_input_mlb_prob.preds, _input_mlb_prob.target, NUM_CLASSES, None, None, _sk_spec),
        (_input_mlb.preds, _input_mlb.target, NUM_CLASSES, False, None, _sk_spec),
        (_input_mcls_prob.preds, _input_mcls_prob.target, NUM_CLASSES, None, None, _sk_spec),
        (_input_mcls.preds, _input_mcls.target, NUM_CLASSES, None, None, _sk_spec),
        (_input_mdmc.preds, _input_mdmc.target, NUM_CLASSES, None, "global", _sk_spec_mdim_mcls),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, NUM_CLASSES, None, "global", _sk_spec_mdim_mcls),
        (_input_mdmc.preds, _input_mdmc.target, NUM_CLASSES, None, "samplewise", _sk_spec_mdim_mcls),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, NUM_CLASSES, None, "samplewise", _sk_spec_mdim_mcls),
    ],
)
class TestSpecificity(MetricTester):

    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_specificity_class(
        self,
        ddp: bool,
        dist_sync_on_step: bool,
        preds: Tensor,
        target: Tensor,
        sk_wrapper: Callable,
        metric_class: Metric,
        metric_fn: Callable,
        multiclass: Optional[bool],
        num_classes: Optional[int],
        average: str,
        mdmc_average: Optional[str],
        ignore_index: Optional[int],
    ):
        # todo: `metric_fn` is unused
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
                reduce=average,
                num_classes=num_classes,
                multiclass=multiclass,
                ignore_index=ignore_index,
                mdmc_reduce=mdmc_average,
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "multiclass": multiclass,
                "ignore_index": ignore_index,
                "mdmc_average": mdmc_average,
            },
            check_dist_sync_on_step=True,
            check_batch=True,
        )

    def test_specificity_fn(
        self,
        preds: Tensor,
        target: Tensor,
        sk_wrapper: Callable,
        metric_class: Metric,
        metric_fn: Callable,
        multiclass: Optional[bool],
        num_classes: Optional[int],
        average: str,
        mdmc_average: Optional[str],
        ignore_index: Optional[int],
    ):
        # todo: `metric_class` is unused
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
                reduce=average,
                num_classes=num_classes,
                multiclass=multiclass,
                ignore_index=ignore_index,
                mdmc_reduce=mdmc_average,
            ),
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "multiclass": multiclass,
                "ignore_index": ignore_index,
                "mdmc_average": mdmc_average,
            },
        )

    def test_accuracy_differentiability(
        self,
        preds: Tensor,
        target: Tensor,
        sk_wrapper: Callable,
        metric_class: Metric,
        metric_fn: Callable,
        multiclass: Optional[bool],
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

        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=metric_class,
            metric_functional=metric_fn,
            metric_args={
                "num_classes": num_classes,
                "average": average,
                "threshold": THRESHOLD,
                "multiclass": multiclass,
                "ignore_index": ignore_index,
                "mdmc_average": mdmc_average,
            }
        )


_mc_k_target = tensor([0, 1, 2])
_mc_k_preds = tensor([[0.35, 0.4, 0.25], [0.1, 0.5, 0.4], [0.2, 0.1, 0.7]])
_ml_k_target = tensor([[0, 1, 0], [1, 1, 0], [0, 0, 0]])
_ml_k_preds = tensor([[0.9, 0.2, 0.75], [0.1, 0.7, 0.8], [0.6, 0.1, 0.7]])


@pytest.mark.parametrize("metric_class, metric_fn", [(Specificity, specificity)])
@pytest.mark.parametrize(
    "k, preds, target, average, expected_spec",
    [
        (1, _mc_k_preds, _mc_k_target, "micro", tensor(5 / 6)),
        (2, _mc_k_preds, _mc_k_target, "micro", tensor(1 / 2)),
        (1, _ml_k_preds, _ml_k_target, "micro", tensor(1 / 2)),
        (2, _ml_k_preds, _ml_k_target, "micro", tensor(1 / 6)),
    ],
)
def test_top_k(
    metric_class,
    metric_fn,
    k: int,
    preds: Tensor,
    target: Tensor,
    average: str,
    expected_spec: Tensor,
):
    """A simple test to check that top_k works as expected.

    Just a sanity check, the tests in StatScores should already guarantee the correctness of results.
    """

    class_metric = metric_class(top_k=k, average=average, num_classes=3)
    class_metric.update(preds, target)

    assert torch.equal(class_metric.compute(), expected_spec)
    assert torch.equal(metric_fn(preds, target, top_k=k, average=average, num_classes=3), expected_spec)


@pytest.mark.parametrize("metric_class, metric_fn", [(Specificity, specificity)])
@pytest.mark.parametrize(
    "ignore_index, expected", [(None, torch.tensor([0.0, np.nan])), (0, torch.tensor([np.nan, np.nan]))]
)
def test_class_not_present(metric_class, metric_fn, ignore_index, expected):
    """This tests that when metric is computed per class and a given class is not present
    in both the `preds` and `target`, the resulting score is `nan`.
    """
    preds = torch.tensor([0, 0, 0])
    target = torch.tensor([0, 0, 0])
    num_classes = 2

    # test functional
    result_fn = metric_fn(preds, target, average=AverageMethod.NONE, num_classes=num_classes, ignore_index=ignore_index)
    assert torch.allclose(expected, result_fn, equal_nan=True)

    # test class
    cl_metric = metric_class(average=AverageMethod.NONE, num_classes=num_classes, ignore_index=ignore_index)
    cl_metric(preds, target)
    result_cl = cl_metric.compute()
    assert torch.allclose(expected, result_cl, equal_nan=True)

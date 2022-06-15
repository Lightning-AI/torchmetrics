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
from typing import Optional

import pytest
from tests.helpers import seed_all
from tests.helpers.testers import NUM_CLASSES, MetricTester
from torch import Tensor, isinf, max, ones_like, reciprocal, tensor, where

# from tests.classification.inputs import _input_multilabel_multidim as _input_mlmd
# from tests.classification.inputs import _input_multilabel_multidim_logits as _input_mlmd_logits
# from tests.classification.inputs import _input_multilabel_multidim_prob as _input_mlmd_prob
from tests.classification.inputs import (  # EXTRA_DIM,
    _input_binary,
    _input_binary_logits,
    _input_binary_multidim,
    _input_binary_multidim_logits,
    _input_binary_multidim_prob,
    _input_binary_prob,
)
from tests.classification.inputs import _input_multiclass as _input_mcls
from tests.classification.inputs import _input_multiclass_logits as _input_mcls_logits
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multiclass_with_missing_class as _input_miss_class
from tests.classification.inputs import _input_multidim_multiclass as _input_mdmc
from tests.classification.inputs import _input_multidim_multiclass_logits as _input_mdmc_logits
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_logits as _input_mlb_logits
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from torchmetrics import GeneralizedDiceScore
from torchmetrics.functional import generalized_dice_score
from torchmetrics.functional.classification.stat_scores import _del_column
from torchmetrics.utilities.checks import _input_format_classification

seed_all(42)


def _sk_generalized_dice(
    preds: Tensor,
    target: Tensor,
    weight_type: str,
    multiclass: bool,
    num_classes: int,
    ignore_index: Optional[int] = None,
    zero_division: Optional[int] = None,
) -> float:
    """Compute generalized dice score from 1D prediction and target.

    Args:
        preds: prediction tensor
        target: target tensor
        weight_type: type of weight to use.
        multiclass: whether problem is multiclass.
        num_classes: number of classes.
        ignore_index: integer specifying a target class to ignore.
        zero_division: The value to use for the score if denominator equals zero. If set to 0, score will be 1
            if the numerator is also 0, and 0 otherwise
    Return:
        Float generalized dice score
    """
    sk_preds, sk_target, mode = _input_format_classification(
        preds, target, multiclass=multiclass, num_classes=num_classes
    )

    if ignore_index is not None:
        sk_preds = _del_column(sk_preds, ignore_index)
        sk_target = _del_column(sk_target, ignore_index)

    # Compute intersection, target and prediction volumes
    intersection = sk_preds * sk_target
    target_volume = sk_target
    pred_volume = sk_preds
    volume = target_volume + pred_volume

    # Reduce over the spatial dimension, if there is one, from (N, C, X) to (N, C)
    if sk_preds.ndim == 3:
        intersection = intersection.sum(dim=2)
        target_volume = target_volume.sum(dim=2)
        pred_volume = pred_volume.sum(dim=2)
        volume = volume.sum(dim=2)

    # Weight computation per sample per class
    if weight_type == "simple":
        weights = reciprocal(target_volume.float())
    elif weight_type == "square":
        weights = reciprocal(target_volume.float() * target_volume.float())
    elif weight_type is None:
        weights = ones_like(target_volume.float())

    # Replace infinites by maximum weight value for the sample. If all weights are infinite, replace by 0
    if weights.dim() > 1:
        for sample_weights in weights:
            infs = isinf(sample_weights)
            sample_weights[infs] = max(sample_weights[~infs]) if len(sample_weights[~infs]) > 0 else 0
    else:
        infs = isinf(weights)
        weights[infs] = max(weights[~infs])

    # Reduce from (N, C) into (N)
    numerator = 2 * (weights * intersection).sum(dim=-1)
    denominator = (weights * volume).sum(dim=-1)
    pred_volume = pred_volume.sum(dim=-1)

    # Compute score and handle zero division
    score = numerator / denominator
    if zero_division is None:
        score = where((denominator == 0) & (pred_volume == 0), tensor(1).float(), score)
        score = where((denominator == 0) & (pred_volume != 0), tensor(0).float(), score)
    else:
        score[denominator == 0] = zero_division

    # Return mean over samples
    return score.mean()


@pytest.mark.parametrize(
    ["pred", "target", "expected"],
    [
        ([[0, 0], [1, 1]], [[0, 0], [1, 1]], 1.0),
        ([[1, 1], [0, 0]], [[0, 0], [1, 1]], 0.0),
        ([[1, 1], [1, 1]], [[1, 1], [0, 0]], 0.5),
        ([[1, 1], [0, 0]], [[1, 1], [0, 0]], 1.0),
    ],
)
def test_generalized_dice_score(pred, target, expected):
    score = generalized_dice_score(tensor(pred), tensor(target))
    assert score == expected


@pytest.mark.parametrize(
    "preds, target, multiclass, multidim, num_classes",
    [
        (_input_binary_multidim.preds, _input_binary_multidim.target, True, True, 2),
        (_input_binary_multidim_logits.preds, _input_binary_multidim_logits.target, True, True, 2),
        (_input_binary_multidim_prob.preds, _input_binary_multidim_prob.target, True, True, 2),
        (_input_binary.preds, _input_binary.target, True, False, 2),
        (_input_binary_logits.preds, _input_binary_logits.target, True, False, 2),
        (_input_binary_prob.preds, _input_binary_prob.target, True, False, 2),
    ],
)
@pytest.mark.parametrize("zero_division", [None, 0, 1])
@pytest.mark.parametrize("ignore_index", [None, 0])
@pytest.mark.parametrize("weight_type", ["simple", "square", None])
class TestGeneralizedDiceBinary(MetricTester):
    @pytest.mark.parametrize("ddp", [False])
    @pytest.mark.parametrize("dist_sync_on_step", [False])
    def test_generalized_dice_class(
        self,
        ddp,
        dist_sync_on_step,
        preds,
        target,
        multiclass,
        multidim,
        weight_type,
        ignore_index,
        num_classes,
        zero_division,
    ):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=GeneralizedDiceScore,
            sk_metric=partial(
                _sk_generalized_dice,
                weight_type=weight_type,
                ignore_index=ignore_index,
                multiclass=multiclass,
                num_classes=num_classes,
                zero_division=zero_division,
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "ignore_index": ignore_index,
                "weight_type": weight_type,
                "multiclass": multiclass,
                "multidim": multidim,
                "num_classes": num_classes,
                "zero_division": zero_division,
            },
        )

    def test_generalized_dice_fn(
        self, preds, target, multiclass, multidim, weight_type, ignore_index, num_classes, zero_division
    ):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=generalized_dice_score,
            sk_metric=partial(
                _sk_generalized_dice,
                weight_type=weight_type,
                ignore_index=ignore_index,
                multiclass=multiclass,
                num_classes=num_classes,
                zero_division=zero_division,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "weight_type": weight_type,
                "multiclass": multiclass,
                "multidim": multidim,
                "num_classes": num_classes,
                "zero_division": zero_division,
            },
        )


@pytest.mark.parametrize(
    "preds, target, multiclass, multidim, num_classes",
    [
        (_input_mcls.preds, _input_mcls.target, True, False, NUM_CLASSES),
        (_input_mcls_logits.preds, _input_mcls_logits.target, True, False, NUM_CLASSES),
        (_input_mcls_prob.preds, _input_mcls_prob.target, True, False, NUM_CLASSES),
        (_input_mdmc.preds, _input_mdmc.target, True, True, NUM_CLASSES),
        (_input_mdmc_logits.preds, _input_mdmc_logits.target, True, True, NUM_CLASSES),
        (_input_mdmc_prob.preds, _input_mdmc_prob.target, True, True, NUM_CLASSES),
        (_input_miss_class.preds, _input_miss_class.target, True, False, NUM_CLASSES),
        (_input_mlb.preds, _input_mlb.target, False, False, NUM_CLASSES),
        (_input_mlb_logits.preds, _input_mlb_logits.target, False, False, NUM_CLASSES),
        (_input_mlb_prob.preds, _input_mlb_prob.target, False, False, NUM_CLASSES),
        # (_input_mlmd.preds, _input_mlmd.target, True, True, NUM_CLASSES),
        # (_input_mlmd_logits.preds, _input_mlmd_logits.target, False, True, NUM_CLASSES * EXTRA_DIM),
        # (_input_mlmd_prob.preds, _input_mlmd_prob.target, True, True, NUM_CLASSES * EXTRA_DIM),
    ],
)
@pytest.mark.parametrize("zero_division", [None, 0, 1])
@pytest.mark.parametrize("ignore_index", [None, 0])
@pytest.mark.parametrize("weight_type", ["simple", "square", None])
class TestGeneralizedDiceMulti(MetricTester):
    @pytest.mark.parametrize("ddp", [False])
    @pytest.mark.parametrize("dist_sync_on_step", [False])
    def test_generalized_dice_class(
        self,
        ddp,
        dist_sync_on_step,
        preds,
        target,
        multiclass,
        multidim,
        weight_type,
        ignore_index,
        num_classes,
        zero_division,
    ):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=GeneralizedDiceScore,
            sk_metric=partial(
                _sk_generalized_dice,
                weight_type=weight_type,
                ignore_index=ignore_index,
                multiclass=multiclass,
                num_classes=num_classes,
                zero_division=zero_division,
            ),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "ignore_index": ignore_index,
                "weight_type": weight_type,
                "multiclass": multiclass,
                "multidim": multidim,
                "num_classes": num_classes,
                "zero_division": zero_division,
            },
        )

    def test_generalized_dice_fn(
        self, preds, target, multiclass, multidim, weight_type, ignore_index, num_classes, zero_division
    ):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=generalized_dice_score,
            sk_metric=partial(
                _sk_generalized_dice,
                weight_type=weight_type,
                ignore_index=ignore_index,
                multiclass=multiclass,
                num_classes=num_classes,
                zero_division=zero_division,
            ),
            metric_args={
                "ignore_index": ignore_index,
                "weight_type": weight_type,
                "multiclass": multiclass,
                "multidim": multidim,
                "num_classes": num_classes,
                "zero_division": zero_division,
            },
        )

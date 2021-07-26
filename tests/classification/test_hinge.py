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
from sklearn.metrics import hinge_loss as sk_hinge
from sklearn.preprocessing import OneHotEncoder

from tests.classification.inputs import Input
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES, MetricTester
from torchmetrics import Hinge
from torchmetrics.functional import hinge
from torchmetrics.functional.classification.hinge import MulticlassMode

torch.manual_seed(42)

_input_binary = Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE), target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_binary_single = Input(preds=torch.randn((NUM_BATCHES, 1)), target=torch.randint(high=2, size=(NUM_BATCHES, 1)))

_input_multiclass = Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
)


def _sk_hinge(preds, target, squared, multiclass_mode):
    sk_preds, sk_target = preds.numpy(), target.numpy()

    if multiclass_mode == MulticlassMode.ONE_VS_ALL:
        enc = OneHotEncoder()
        enc.fit(sk_target.reshape(-1, 1))
        sk_target = enc.transform(sk_target.reshape(-1, 1)).toarray()

    if sk_preds.ndim == 1 or multiclass_mode == MulticlassMode.ONE_VS_ALL:
        sk_target = 2 * sk_target - 1

    if squared or sk_target.max() != 1 or sk_target.min() != -1:
        # Squared not an option in sklearn and infers classes incorrectly with single element, so adapted from source
        if sk_preds.ndim == 1 or multiclass_mode == MulticlassMode.ONE_VS_ALL:
            margin = sk_target * sk_preds
        else:
            mask = np.ones_like(sk_preds, dtype=bool)
            mask[np.arange(sk_target.shape[0]), sk_target] = False
            margin = sk_preds[~mask]
            margin -= np.max(sk_preds[mask].reshape(sk_target.shape[0], -1), axis=1)
        measures = 1 - margin
        measures = np.clip(measures, 0, None)

        if squared:
            measures = measures**2
        return measures.mean(axis=0)
    if multiclass_mode == MulticlassMode.ONE_VS_ALL:
        result = np.zeros(sk_preds.shape[1])
        for i in range(result.shape[0]):
            result[i] = sk_hinge(y_true=sk_target[:, i], pred_decision=sk_preds[:, i])
        return result

    return sk_hinge(y_true=sk_target, pred_decision=sk_preds)


@pytest.mark.parametrize(
    "preds, target, squared, multiclass_mode",
    [
        (_input_binary.preds, _input_binary.target, False, None),
        (_input_binary.preds, _input_binary.target, True, None),
        (_input_binary_single.preds, _input_binary_single.target, False, None),
        (_input_binary_single.preds, _input_binary_single.target, True, None),
        (_input_multiclass.preds, _input_multiclass.target, False, MulticlassMode.CRAMMER_SINGER),
        (_input_multiclass.preds, _input_multiclass.target, True, MulticlassMode.CRAMMER_SINGER),
        (_input_multiclass.preds, _input_multiclass.target, False, MulticlassMode.ONE_VS_ALL),
        (_input_multiclass.preds, _input_multiclass.target, True, MulticlassMode.ONE_VS_ALL),
    ],
)
class TestHinge(MetricTester):

    @pytest.mark.parametrize("ddp", [True, False])
    @pytest.mark.parametrize("dist_sync_on_step", [True, False])
    def test_hinge_class(self, ddp, dist_sync_on_step, preds, target, squared, multiclass_mode):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=Hinge,
            sk_metric=partial(_sk_hinge, squared=squared, multiclass_mode=multiclass_mode),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "squared": squared,
                "multiclass_mode": multiclass_mode,
            },
        )

    def test_hinge_fn(self, preds, target, squared, multiclass_mode):
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=partial(hinge, squared=squared, multiclass_mode=multiclass_mode),
            sk_metric=partial(_sk_hinge, squared=squared, multiclass_mode=multiclass_mode),
        )

    def test_hinge_differentiability(self, preds, target, squared, multiclass_mode):
        self.run_differentiability_test(
            preds=preds,
            target=target,
            metric_module=Hinge,
            metric_functional=partial(hinge, squared=squared, multiclass_mode=multiclass_mode)
        )


_input_multi_target = Input(preds=torch.randn(BATCH_SIZE), target=torch.randint(high=2, size=(BATCH_SIZE, 2)))

_input_binary_different_sizes = Input(
    preds=torch.randn(BATCH_SIZE * 2), target=torch.randint(high=2, size=(BATCH_SIZE, ))
)

_input_multi_different_sizes = Input(
    preds=torch.randn(BATCH_SIZE * 2, NUM_CLASSES), target=torch.randint(high=NUM_CLASSES, size=(BATCH_SIZE, ))
)

_input_extra_dim = Input(
    preds=torch.randn(BATCH_SIZE, NUM_CLASSES, 2), target=torch.randint(high=2, size=(BATCH_SIZE, ))
)


@pytest.mark.parametrize(
    "preds, target, multiclass_mode",
    [(_input_multi_target.preds, _input_multi_target.target, None),
     (_input_binary_different_sizes.preds, _input_binary_different_sizes.target, None),
     (_input_multi_different_sizes.preds, _input_multi_different_sizes.target, None),
     (_input_extra_dim.preds, _input_extra_dim.target, None),
     (_input_multiclass.preds[0], _input_multiclass.target[0], 'invalid_mode')],
)
def test_bad_inputs_fn(preds, target, multiclass_mode):
    with pytest.raises(ValueError):
        _ = hinge(preds, target, multiclass_mode=multiclass_mode)


def test_bad_inputs_class():
    with pytest.raises(ValueError):
        Hinge(multiclass_mode='invalid_mode')

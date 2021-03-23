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
from sklearn.metrics import hinge_loss as sk_hinge_loss
from sklearn.preprocessing import OneHotEncoder

from tests.classification.inputs import Input
from tests.helpers.testers import BATCH_SIZE, NUM_BATCHES, NUM_CLASSES, MetricTester
from torchmetrics import HingeLoss
from torchmetrics.functional import hinge_loss

torch.manual_seed(42)


_input_binary = Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE),
    target=torch.randint(high=2, size=(NUM_BATCHES, BATCH_SIZE))
)

_input_multiclass = Input(
    preds=torch.randn(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))
)


def _sk_hinge_loss(preds, target, squared, multiclass_mode):
    sk_preds, sk_target = preds.numpy(), target.numpy()

    if multiclass_mode == 'one_vs_all':
        enc = OneHotEncoder()
        enc.fit(sk_target.reshape(-1, 1))
        sk_target = enc.transform(sk_target.reshape(-1, 1)).toarray()

    if sk_preds.ndim == 1 or multiclass_mode == 'one_vs_all':
        sk_target = 2 * sk_target - 1

    if squared:  # Squared not an option in sklearn, so adapted from source
        if sk_preds.ndim == 1 or multiclass_mode == 'one_vs_all':
            margin = sk_target * sk_preds
        else:
            mask = np.ones_like(sk_preds, dtype=bool)
            mask[np.arange(sk_target.shape[0]), sk_target] = False
            margin = sk_preds[~mask]
            margin -= np.max(sk_preds[mask].reshape(sk_target.shape[0], -1), axis=1)
        losses = 1 - margin
        np.clip(losses, 0, None, out=losses)
        losses = losses ** 2
        return losses.mean(axis=0)
    else:
        if multiclass_mode == 'one_vs_all':
            result = np.zeros(sk_preds.shape[1])
            for i in range(result.shape[0]):
                result[i] = sk_hinge_loss(y_true=sk_target[:, i], pred_decision=sk_preds[:, i])
            return result

        return sk_hinge_loss(y_true=sk_target, pred_decision=sk_preds)


@pytest.mark.parametrize(
    "preds, target, squared, multiclass_mode",
    [
        (_input_binary.preds, _input_binary.target, False, None),
        (_input_binary.preds, _input_binary.target, True, None),
        (_input_multiclass.preds, _input_multiclass.target, False, 'crammer_singer'),
        (_input_multiclass.preds, _input_multiclass.target, True, 'crammer_singer'),
        (_input_multiclass.preds, _input_multiclass.target, False, 'one_vs_all'),
        (_input_multiclass.preds, _input_multiclass.target, True, 'one_vs_all'),
    ],
)
class TestHingeLoss(MetricTester):

    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_hinge_loss_class(self, ddp, dist_sync_on_step, preds, target, squared, multiclass_mode):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=HingeLoss,
            sk_metric=partial(_sk_hinge_loss, squared=squared, multiclass_mode=multiclass_mode),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "squared": squared,
                "multiclass_mode": multiclass_mode,
            },
        )

    def test_hinge_loss_fn(self, preds, target, squared, multiclass_mode):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=partial(hinge_loss, squared=squared, multiclass_mode=multiclass_mode),
            sk_metric=partial(_sk_hinge_loss, squared=squared, multiclass_mode=multiclass_mode),
        )


_input_multi_target = Input(
    preds=torch.randn(BATCH_SIZE),
    target=torch.randint(high=2, size=(BATCH_SIZE, 2))
)

_input_binary_different_sizes = Input(
    preds=torch.randn(BATCH_SIZE * 2),
    target=torch.randint(high=2, size=(BATCH_SIZE,))
)

_input_multi_different_sizes = Input(
    preds=torch.randn(BATCH_SIZE * 2, NUM_CLASSES),
    target=torch.randint(high=NUM_CLASSES, size=(BATCH_SIZE,))
)

_input_extra_dim = Input(
    preds=torch.randn(BATCH_SIZE, NUM_CLASSES, 2),
    target=torch.randint(high=2, size=(BATCH_SIZE,))
)


@pytest.mark.parametrize(
    "preds, target, multiclass_mode",
    [
        (_input_multi_target.preds, _input_multi_target.target, None),
        (_input_binary_different_sizes.preds, _input_binary_different_sizes.target, None),
        (_input_multi_different_sizes.preds, _input_multi_different_sizes.target, None),
        (_input_extra_dim.preds, _input_extra_dim.target, None),
        (_input_multiclass.preds[0], _input_multiclass.target[0], 'invalid_mode')
    ],
)
def test_bad_inputs(preds, target, multiclass_mode):
    with pytest.raises(ValueError):
        _ = hinge_loss(preds, target, multiclass_mode=multiclass_mode)

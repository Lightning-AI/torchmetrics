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


def _sk_hinge_loss(preds, target, squared):
    sk_preds, sk_target = preds.numpy(), target.numpy()

    if sk_preds.ndim == 1:
        sk_target = 2 * sk_target - 1

    if squared:  # Squared not an option in sklearn, so adapted from source
        if sk_preds.ndim == 1:
            margin = sk_target * sk_preds
        else:
            mask = np.ones_like(sk_preds, dtype=bool)
            mask[np.arange(sk_target.shape[0]), sk_target] = False
            margin = sk_preds[~mask]
            margin -= np.max(sk_preds[mask].reshape(sk_target.shape[0], -1), axis=1)
        losses = 1 - margin
        np.clip(losses, 0, None, out=losses)
        losses = losses ** 2
        return losses.mean()

    return sk_hinge_loss(y_true=sk_target, pred_decision=sk_preds)


@pytest.mark.parametrize(
    "preds, target, squared",
    [
        (_input_binary.preds, _input_binary.target, False),
        (_input_binary.preds, _input_binary.target, True),
        (_input_multiclass.preds, _input_multiclass.target, False),
        (_input_multiclass.preds, _input_multiclass.target, True),
    ],
)
class TestHingeLoss(MetricTester):

    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_hinge_loss_class(self, ddp, dist_sync_on_step, preds, target, squared):
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=HingeLoss,
            sk_metric=partial(_sk_hinge_loss, squared=squared),
            dist_sync_on_step=dist_sync_on_step,
            metric_args={
                "squared": squared
            },
        )

    def test_hinge_loss_fn(self, preds, target, squared):
        self.run_functional_metric_test(
            preds,
            target,
            metric_functional=partial(hinge_loss, squared=squared),
            sk_metric=partial(_sk_hinge_loss, squared=squared),
        )

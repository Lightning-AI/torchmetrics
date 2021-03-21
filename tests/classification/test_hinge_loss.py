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
from torch import tensor

from tests.classification.inputs import _input_binary, _input_binary_prob
from tests.classification.inputs import _input_multiclass as _input_mcls
from tests.classification.inputs import _input_multiclass_prob as _input_mcls_prob
from tests.classification.inputs import _input_multidim_multiclass as _input_mdmc
from tests.classification.inputs import _input_multidim_multiclass_prob as _input_mdmc_prob
from tests.classification.inputs import _input_multilabel as _input_mlb
from tests.classification.inputs import _input_multilabel_multidim as _input_mlmd
from tests.classification.inputs import _input_multilabel_multidim_prob as _input_mlmd_prob
from tests.classification.inputs import _input_multilabel_prob as _input_mlb_prob
from tests.helpers.testers import THRESHOLD, MetricTester
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy
from torchmetrics.functional import hinge_loss
from torchmetrics.utilities.checks import _input_format_classification
from torchmetrics.utilities.enums import DataType

torch.manual_seed(42)


def _sk_hinge_loss(preds, target):
    sk_preds, sk_target = preds.numpy(), target.numpy()

    return sk_hinge_loss(y_true=sk_target, pred_decision=sk_preds)


def test_hinge_loss():
    pred = tensor([[-2.5, 1.2, 0.1], [-0.6, 1.3, 1.5], [-3.2, 0.8, -1.7]])
    target = tensor([0, 2, 1])

    sk_res = _sk_hinge_loss(pred, target)
    tm_res = hinge_loss(pred, target)

    assert torch.allclose(torch.tensor(sk_res), tm_res)

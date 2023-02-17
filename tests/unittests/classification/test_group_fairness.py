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
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from scipy.special import expit as sigmoid

from torchmetrics.classification.group_fairness import BinaryFairness
from torchmetrics.functional.classification.group_fairness import binary_fairness
from unittests.classification.inputs import _group_cases
from unittests.helpers import seed_all
from unittests.helpers.testers import THRESHOLD, MetricTester, inject_ignore_index, remove_ignore_index_groups

seed_all(42)


def _fairlearn_binary(preds, target, groups, ignore_index):
    metrics = {"dp": selection_rate, "eo": true_positive_rate}

    preds = preds.numpy()
    target = target.numpy()
    groups = groups.numpy()

    if np.issubdtype(preds.dtype, np.floating):
        if not ((preds > 0) & (preds < 1)).all():
            preds = sigmoid(preds)
        preds = (preds >= THRESHOLD).astype(np.uint8)

    target, preds, groups = remove_ignore_index_groups(target, preds, groups, ignore_index)

    mf = MetricFrame(metrics=metrics, y_true=target, y_pred=preds, sensitive_features=groups)

    mf_group = mf.by_group
    ratios = mf.ratio()

    return {
        f"DP_{mf_group['dp'].idxmin()}_{mf_group['dp'].idxmax()}": torch.tensor(ratios["dp"], dtype=torch.float),
        f"EO_{mf_group['eo'].idxmin()}_{mf_group['eo'].idxmax()}": torch.tensor(ratios["eo"], dtype=torch.float),
    }


@pytest.mark.parametrize("input", _group_cases)
class TestBinaryFairness(MetricTester):
    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    @pytest.mark.parametrize("ddp", [False, True])
    def test_binary_fairness(self, ddp, input, ignore_index):
        preds, target, groups = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=BinaryFairness,
            reference_metric=partial(_fairlearn_binary, ignore_index=ignore_index),
            metric_args={"threshold": THRESHOLD, "ignore_index": ignore_index, "num_groups": 2, "task": "all"},
            groups=groups,
            fragment_kwargs=True,
        )

    @pytest.mark.parametrize("ignore_index", [None, 0, -1])
    def test_binary_fairness_functional(self, input, ignore_index):
        preds, target, groups = input
        if ignore_index == -1:
            target = inject_ignore_index(target, ignore_index)

        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=binary_fairness,
            reference_metric=partial(_fairlearn_binary, ignore_index=ignore_index),
            metric_args={
                "threshold": THRESHOLD,
                "ignore_index": ignore_index,
                "task": "all",
            },
            groups=groups,
            fragment_kwargs=True,
        )

    # def test_binary_fairness_differentiability(self, input):
    #     preds, target, groups = input
    #     self.run_differentiability_test(
    #         preds=preds,
    #         target=target,
    #         metric_module=BinaryFairness,
    #         metric_functional=binary_fairness,
    #         metric_args={"threshold": THRESHOLD, "task": "all"},
    #         groups=groups,
    #         fragment_kwargs=True,
    #     )

    # @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    # def test_binary_fairness_half_cpu(self, input, dtype):
    #     preds, target, groups = input

    #     if (preds < 0).any() and dtype == torch.half:
    #         pytest.xfail(reason="torch.sigmoid in metric does not support cpu + half precision")
    #     self.run_precision_test_cpu(
    #         preds=preds,
    #         target=target,
    #         metric_module=BinaryFairness,
    #         metric_functional=binary_fairness,
    #         metric_args={"threshold": THRESHOLD, "num_groups": 2, "task": "all"},
    #         dtype=dtype,
    #         groups=groups,
    #     )

    # @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    # @pytest.mark.parametrize("dtype", [torch.half, torch.double])
    # def test_binary_fairness_half_gpu(self, input, dtype):
    #     preds, target, groups = input
    #     self.run_precision_test_gpu(
    #         preds=preds,
    #         target=target,
    #         metric_module=BinaryFairness,
    #         metric_functional=binary_fairness,
    #         metric_args={"threshold": THRESHOLD, "num_groups": 2, "task": "all"},
    #         dtype=dtype,
    #         groups=groups,
    #     )

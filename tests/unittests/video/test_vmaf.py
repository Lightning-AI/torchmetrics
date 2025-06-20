# Copyright The Lightning team.
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

import pandas as pd  # pandas is installed as a dependency of vmaf-torch
import pytest
import torch
from einops import rearrange
from vmaf_torch import VMAF

from torchmetrics.functional.video.vmaf import calculate_luma, video_multi_method_assessment_fusion
from torchmetrics.utilities.imports import _TORCH_VMAF_AVAILABLE
from torchmetrics.video import VideoMultiMethodAssessmentFusion
from unittests import _Input
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester

seed_all(42)


def _reference_vmaf_no_features(preds, target) -> dict[str, torch.Tensor] | torch.Tensor:
    """Reference implementation of VMAF metric.

    This should preferably be replaced with the python version of the netflix library
    https://github.com/Netflix/vmaf
    but that requires it to be compiled on the system.

    """
    b = preds.shape[0]
    orig_dtype, device = preds.dtype, preds.device
    preds_luma = calculate_luma(preds)
    target_luma = calculate_luma(target)
    vmaf = VMAF().to(device)

    # we need to compute the model for each video separately
    scores = [
        vmaf.compute_vmaf_score(
            rearrange(target_luma[video], "c f h w -> f c h w"), rearrange(preds_luma[video], "c f h w -> f c h w")
        )
        for video in range(b)
    ]
    return torch.cat(scores, dim=1).t().to(orig_dtype)


def _reference_vmaf_with_features(preds, target) -> dict[str, torch.Tensor] | torch.Tensor:
    """Reference implementation of VMAF metric.

    This should preferably be replaced with the python version of the netflix library
    https://github.com/Netflix/vmaf
    but that requires it to be compiled on the system.

    """
    b = preds.shape[0]
    orig_dtype, device = preds.dtype, preds.device
    preds_luma = calculate_luma(preds)
    target_luma = calculate_luma(target)
    vmaf = VMAF().to(device)

    scores_and_features = [
        vmaf.table(
            rearrange(target_luma[video], "c f h w -> f c h w"), rearrange(preds_luma[video], "c f h w -> f c h w")
        )
        for video in range(b)
    ]
    dfs = [scores_and_features[video].apply(pd.to_numeric, errors="coerce") for video in range(b)]
    result = [
        {col: torch.tensor(dfs[video][col].values, dtype=orig_dtype) for col in dfs[video].columns if col != "Frame"}
        for video in range(b)
    ]
    return {col: torch.stack([result[video][col] for video in range(b)]) for col in result[0]}


# Define inputs
NUM_BATCHES, BATCH_SIZE, FRAMES = 2, 4, 10
_inputs = []
for size in [32, 64]:
    preds = torch.rand(NUM_BATCHES, BATCH_SIZE, 3, FRAMES, size, size)
    target = torch.rand(NUM_BATCHES, BATCH_SIZE, 3, FRAMES, size, size)
    _inputs.append(_Input(preds=preds, target=target))


@pytest.mark.skipif(not _TORCH_VMAF_AVAILABLE, reason="test requires vmaf-torch")
@pytest.mark.parametrize(("preds", "target"), [(i.preds, i.target) for i in _inputs])
@pytest.mark.parametrize("features", [True, False])
class TestVMAF(MetricTester):
    """Test class for `VideoMultiMethodAssessmentFusion` metric."""

    atol = 1e-3

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_vmaf_module(self, preds, target, features, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=VideoMultiMethodAssessmentFusion,
            reference_metric=partial(_reference_vmaf_with_features if features else _reference_vmaf_no_features),
            metric_args={"features": features},
        )

    def test_vmaf_functional(self, preds, target, features):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=video_multi_method_assessment_fusion,
            reference_metric=partial(_reference_vmaf_with_features if features else _reference_vmaf_no_features),
            metric_args={"features": features},
        )

    def test_vmaf_features_shape(self, preds, target, features):
        """Test that the shape of the features is correct."""
        if not features:
            return
        vmaf_dict = video_multi_method_assessment_fusion(preds[0], target[0], features=features)
        for key in vmaf_dict:
            assert vmaf_dict[key].shape == (BATCH_SIZE, FRAMES), (
                f"Shape of {key} is incorrect. Expected {(BATCH_SIZE, FRAMES)}, got {vmaf_dict[key].shape}"
            )


def test_vmaf_raises_error(monkeypatch):
    """Test that the appropriate error is raised when vmaf-torch is not installed."""
    # mock/fake that vmaf-torch is not installed
    monkeypatch.setattr("torchmetrics.functional.video.vmaf._TORCH_VMAF_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="vmaf-torch is not installed"):
        video_multi_method_assessment_fusion(torch.rand(1, 3, 10, 32, 32), torch.rand(1, 3, 10, 32, 32))

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

import pytest
import torch
from torch import Tensor
from torchmetrics.audio import SpeechReverberationModulationEnergyRatio
from torchmetrics.functional.audio import speech_reverberation_modulation_energy_ratio
from srmrpy import srmr as srmrpy_srmr

from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

preds = torch.rand(1, 2, 8000)


def _ref_metric_batch(preds: Tensor, fs: int, fast: bool, norm: bool, **kwargs):
    # shape: preds [BATCH_SIZE, Time]
    shape = preds.shape
    if len(shape) == 1:
        preds = preds.reshape(1, -1)  # [B, time]
    else:
        preds = preds.reshape(-1, shape[-1])  # [B, time]
    n_batch, time = preds.shape

    preds = preds.detach().cpu().numpy()
    score = []
    for b in range(preds.shape[0]):
        val, _ = srmrpy_srmr(preds[b, ...], fs=fs, fast=fast, norm=norm)
        score.append(val)
    score = torch.tensor(score)
    score = score.reshape(*shape[:-1])
    return score


def _average_metric(preds, target, metric_func):
    # shape: preds [BATCH_SIZE, 1, Time] , target [BATCH_SIZE, 1, Time]
    # or shape: preds [NUM_BATCHES*BATCH_SIZE, 1, Time] , target [NUM_BATCHES*BATCH_SIZE, 1, Time]
    return metric_func(preds).mean()


def speech_reverberation_modulation_energy_ratio_cheat(preds, target, **kwargs):
    # cheat the MetricTester as the speech_reverberation_modulation_energy_ratio doesn't need target
    return speech_reverberation_modulation_energy_ratio(preds, **kwargs)


class SpeechReverberationModulationEnergyRatioCheat(SpeechReverberationModulationEnergyRatio):
    # cheat the MetricTester as SpeechReverberationModulationEnergyRatioCheat doesn't need target
    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds=preds)


@pytest.mark.parametrize(
    "preds, fs, fast, norm",
    [
        (preds, 8000, False, False),
        (preds, 8000, False, True),
        (preds, 8000, True, False),
        # (preds, 8000, True, True),
        (preds, 16000, False, False),
        # (preds, 16000, False, True),
        # (preds, 16000, True, False),
        # (preds, 16000, True, True),
    ],
)
class TestSRMR(MetricTester):
    """Test class for `SpeechReverberationModulationEnergyRatio` metric."""

    atol = 1e-2

    @pytest.mark.parametrize("ddp", [True, False])
    def test_srmr(self, preds, fs, fast, norm, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp,
            preds=preds,
            target=preds,
            metric_class=SpeechReverberationModulationEnergyRatioCheat,
            reference_metric=partial(_average_metric, metric_func=_ref_metric_batch),
            metric_args={"fs": fs, "fast": fast, "norm": norm},
        )

    def test_srmr_functional(self, preds, fs, fast, norm):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=preds,
            metric_functional=speech_reverberation_modulation_energy_ratio_cheat,
            reference_metric=_ref_metric_batch,
            metric_args={"fs": fs, "fast": fast, "norm": norm},
        )

    def test_srmr_differentiability(self, preds, fs, fast, norm):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        if fast is True:
            pytest.xfail("SRMR metric is not differentiable when `fast=True`")

        self.run_differentiability_test(
            preds=preds,
            target=preds,
            metric_module=SpeechReverberationModulationEnergyRatioCheat,
            metric_functional=speech_reverberation_modulation_energy_ratio_cheat,
            metric_args={"fs": fs, "fast": fast, "norm": norm},
        )

    def test_srmr_half_cpu(self, preds, fs, fast, norm):
        """Test dtype support of the metric on CPU."""
        pytest.xfail("SRMR metric does not support cpu + half precision")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
    def test_srmr_half_gpu(self, preds, fs, fast, norm):
        """Test dtype support of the metric on GPU."""
        self.run_precision_test_gpu(
            preds=preds,
            target=preds,
            metric_module=SpeechReverberationModulationEnergyRatioCheat,
            metric_functional=speech_reverberation_modulation_energy_ratio_cheat,
            metric_args={"fs": fs, "fast": fast, "norm": norm},
        )

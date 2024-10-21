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
from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor
from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment
from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment
from torchmetrics.utilities.imports import _LIBROSA_AVAILABLE

from unittests._helpers.testers import MetricTester

if _LIBROSA_AVAILABLE:
    import librosa
else:
    librosa = None

# reference values below were calculated using the method described in https://github.com/gabrielmittag/NISQA/blob/master/README.md
inputs = [
    # uniform noise
    {
        "preds": torch.rand(2, 2, 16000, generator=torch.Generator().manual_seed(42)),
        "fs": 16000,
        "reference": torch.tensor([
            [
                [0.8105150461, 1.8459059000, 2.4780223370, 1.0402423143, 1.5687377453],
                [0.8629049063, 1.7767801285, 2.3915612698, 1.0460783243, 1.6212222576],
            ],
            [
                [0.8608418703, 1.9113740921, 2.5213730335, 1.0900889635, 1.6314117908],
                [0.8071692586, 1.7834275961, 2.4235677719, 1.0236976147, 1.5617829561],
            ],
        ]),
    },
    {
        "preds": torch.rand(2, 2, 48000, generator=torch.Generator().manual_seed(42)),
        "fs": 48000,
        "reference": torch.tensor([
            [
                [0.7670641541, 1.1634330750, 2.6056811810, 1.4002652168, 1.5218108892],
                [0.7974857688, 1.1845922470, 2.6476621628, 1.4282002449, 1.5324314833],
            ],
            [
                [0.8114687800, 1.1764185429, 2.6281285286, 1.4396891594, 1.5460423231],
                [0.6779640913, 1.1818346977, 2.5106279850, 1.2842310667, 1.4014176130],
            ],
        ]),
    },
    # 440 Hz tone
    {
        "preds": torch.sin(2 * 3.14159 * 440 / 16000 * torch.arange(16000)),
        "fs": 16000,
        "reference": torch.tensor([1.1243989468, 2.1237702370, 3.6184809208, 1.2584471703, 1.8518198729]),
    },
    {
        "preds": torch.sin(2 * 3.14159 * 440 / 48000 * torch.arange(48000)),
        "fs": 48000,
        "reference": torch.tensor([1.1263639927, 2.1246092319, 3.6191856861, 1.2572505474, 1.8531025648]),
    },
    # 1 kHz square wave
    {
        "preds": torch.sign(torch.sin(2 * 3.14159 * 1000 / 16000 * torch.arange(16000))),
        "fs": 16000,
        "reference": torch.tensor([1.1472485065, 1.5280672312, 3.3269913197, 1.0594099760, 1.8283343315]),
    },
    {
        "preds": torch.sign(torch.sin(2 * 3.14159 * 1000 / 48000 * torch.arange(48000))),
        "fs": 48000,
        "reference": torch.tensor([1.1762716770, 1.7309110165, 3.5088183880, 1.1177743673, 1.8077727556]),
    },
]


def _reference_metric_batch(preds, target, mean):
    def _reference_metric(preds):
        for pred, ref in zip(*[
            [x for i in inputs for x in i[which].reshape(-1, i[which].shape[-1])] for which in ["preds", "reference"]
        ]):
            if torch.equal(preds, pred):
                return ref
        raise NotImplementedError

    out = torch.stack([_reference_metric(pred) for pred in preds.reshape(-1, preds.shape[-1])])
    out = out.reshape(*preds.shape[:-1], 5)
    if mean:
        out = out.reshape(-1, 5).mean(dim=0)
    return out


def _nisqa_cheat(preds, target, **kwargs: Dict[str, Any]):
    # cheat the MetricTester as non_intrusive_speech_quality_assessment does not need a target
    return non_intrusive_speech_quality_assessment(preds, **kwargs)


class _NISQACheat(NonIntrusiveSpeechQualityAssessment):
    # cheat the MetricTester as NonIntrusiveSpeechQualityAssessment does not need a target
    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds=preds)


@pytest.mark.parametrize(
    "preds, fs, reference",
    [(inputs[i]["preds"], inputs[i]["fs"], inputs[i]["reference"]) for i in range(len(inputs))],
)
class TestNISQA(MetricTester):
    """Test class for `NonIntrusiveSpeechQualityAssessment` metric."""

    atol = 5e-5

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_nisqa(self, preds: Tensor, reference: Tensor, fs: int, ddp: bool, device=None):
        """Test class implementation of metric."""
        if preds.ndim == 1:
            preds = preds.unsqueeze(0)
        self.run_class_metric_test(
            ddp,
            preds=preds,
            target=preds,
            metric_class=_NISQACheat,
            reference_metric=partial(_reference_metric_batch, mean=True),
            metric_args={"fs": fs},
        )

    def test_nisqa_functional(self, preds: Tensor, reference: Tensor, fs: int, device="cpu"):
        """Test functional implementation of metric."""
        if preds.ndim == 1:
            preds = preds.unsqueeze(0)
        # double preds because MetricTester.run_functional_metric_test only iterates over num_batches // 2
        preds = torch.cat([preds, preds], dim=0)
        self.run_functional_metric_test(
            preds=preds,
            target=preds,
            metric_functional=_nisqa_cheat,
            reference_metric=partial(_reference_metric_batch, mean=False),
            metric_args={"fs": fs},
        )


@pytest.mark.parametrize("shape", [(3000,), (2, 3000), (1, 2, 3000), (2, 3, 1, 3000)])
def test_shape(shape: Tuple[int]):
    """Test output shape."""
    preds = torch.rand(*shape)
    out = non_intrusive_speech_quality_assessment(preds, 16000)
    assert out.shape == (*shape[:-1], 5)
    metric = NonIntrusiveSpeechQualityAssessment(16000)
    out = metric(preds)
    assert out.shape == (5,)


def test_error_on_short_input():
    """Test error on short input."""
    preds = torch.rand(3000)
    non_intrusive_speech_quality_assessment(preds, 16000)
    with pytest.raises(RuntimeError, match="Input signal is too short."):
        non_intrusive_speech_quality_assessment(preds, 48000)
    preds = torch.rand(2000)
    with pytest.raises(RuntimeError, match="Input signal is too short."):
        non_intrusive_speech_quality_assessment(preds, 16000)
    with pytest.raises(RuntimeError, match="Input signal is too short."):
        non_intrusive_speech_quality_assessment(preds, 48000)


def test_error_on_long_input():
    """Test error on long input."""
    preds = torch.rand(834240)
    with pytest.raises(RuntimeError, match="Maximum number of melspectrogram segments exceeded. Use shorter audio."):
        non_intrusive_speech_quality_assessment(preds, 16000)
    non_intrusive_speech_quality_assessment(preds, 48000)
    preds = torch.rand(2502720)
    with pytest.raises(RuntimeError, match="Maximum number of melspectrogram segments exceeded. Use shorter audio."):
        non_intrusive_speech_quality_assessment(preds, 16000)
    with pytest.raises(RuntimeError, match="Maximum number of melspectrogram segments exceeded. Use shorter audio."):
        non_intrusive_speech_quality_assessment(preds, 48000)

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
        "preds": torch.rand(2, 2, 16000, generator=torch.Generator().manual_seed(42)),
        "fs": 48000,
        "reference": torch.tensor([
            [
                [0.7717766166, 1.1642380953, 2.5894179344, 1.4037175179, 1.4931157827],
                [0.7730888128, 1.2189327478, 2.6117930412, 1.3941351175, 1.5014400482],
            ],
            [
                [0.7035500407, 1.1133824587, 2.5026409626, 1.3385921717, 1.4153599739],
                [0.8563135266, 1.1975537539, 2.7270953655, 1.5784986019, 1.5962394476],
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
        while preds.ndim < 3:
            preds = preds.unsqueeze(0)
        while reference.ndim < 3:
            reference = reference.unsqueeze(0)
        # cheat MetricTester by creating an iterator to retrieve correct reference value
        # reference_metric is called once for each item in the batch and one last time on the whole batch
        _ref_iter = iter([*reference, reference.mean(dim=0)])
        self.run_class_metric_test(
            ddp,
            preds=preds,
            target=preds,
            metric_class=_NISQACheat,
            reference_metric=lambda x, y: next(_ref_iter).mean(dim=0),
            metric_args={"fs": fs},
        )

    def test_nisqa_functional(self, preds: Tensor, reference: Tensor, fs: int, device="cpu"):
        """Test functional implementation of metric."""
        while preds.ndim < 3:
            preds = preds.unsqueeze(0)
        while reference.ndim < 3:
            reference = reference.unsqueeze(0)
        # cheat MetricTester by creating an iterator to retrieve correct reference value
        # reference_metric is called once for each item in the batch
        _ref_iter = iter(reference)
        self.run_functional_metric_test(
            preds=preds,
            target=preds,
            metric_functional=_nisqa_cheat,
            reference_metric=lambda x, y: next(_ref_iter),
            metric_args={"fs": fs},
        )


@pytest.mark.parametrize("shape", [(3000,), (2, 3000), (1, 2, 3000), (2, 3, 1, 3000)])
def test_shape(shape: Tuple[int]):
    preds = torch.rand(*shape)
    out = non_intrusive_speech_quality_assessment(preds, 16000)
    assert out.shape == (*shape[:-1], 5)
    metric = NonIntrusiveSpeechQualityAssessment(16000)
    out = metric(preds)
    assert out.shape == (5,)


def test_error_on_short_input():
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
    preds = torch.rand(834240)
    with pytest.raises(RuntimeError, match="Maximum number of melspectrogram segments exceeded. Use shorter audio."):
        non_intrusive_speech_quality_assessment(preds, 16000)
    non_intrusive_speech_quality_assessment(preds, 48000)
    preds = torch.rand(2502720)
    with pytest.raises(RuntimeError, match="Maximum number of melspectrogram segments exceeded. Use shorter audio."):
        non_intrusive_speech_quality_assessment(preds, 16000)
    with pytest.raises(RuntimeError, match="Maximum number of melspectrogram segments exceeded. Use shorter audio."):
        non_intrusive_speech_quality_assessment(preds, 48000)

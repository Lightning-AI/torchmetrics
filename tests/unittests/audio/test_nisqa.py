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
from typing import Any

import pytest
import torch
from torch import Tensor

from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment
from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment
from unittests._helpers.testers import MetricTester

# reference values below were calculated using the method described in https://github.com/gabrielmittag/NISQA/blob/master/README.md
inputs = [
    {
        "preds": torch.rand(2, 2, 16000, generator=torch.Generator().manual_seed(42)),  # uniform noise
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
        "preds": torch.rand(2, 2, 48000, generator=torch.Generator().manual_seed(42)),  # uniform noise
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
    {
        "preds": torch.stack([
            torch.stack([
                torch.sin(2 * 3.14159 * 440 / 16000 * torch.arange(16000)),  # 1 s 440 Hz tone @ 16 kHz
                torch.sin(2 * 3.14159 * 1000 / 16000 * torch.arange(16000)),  # 1 s 1000 Hz tone @ 16 kHz
            ]),
            torch.stack([
                torch.sign(torch.sin(2 * 3.14159 * 200 / 16000 * torch.arange(16000))),  # 1 s 200 Hz square @ 16 kHz
                (1 + 2 * 200 / 16000 * torch.arange(16000)) % 2 - 1,  # 1 s 200 Hz sawtooth @ 16 kHz
            ]),
        ]),
        "fs": 16000,
        "reference": torch.tensor([
            [
                [1.1243989468, 2.1237702370, 3.6184809208, 1.2584471703, 1.8518198729],
                [1.2761806250, 1.8802671432, 3.3731021881, 1.2554246187, 1.6879540682],
            ],
            [
                [0.9259074330, 2.7644648552, 3.1585879326, 1.4163932800, 1.5672523975],
                [0.8493731022, 2.6398222446, 3.0776870251, 1.1348335743, 1.6034533978],
            ],
        ]),
    },
    {
        "preds": torch.stack([
            torch.stack([
                torch.sin(2 * 3.14159 * 440 / 48000 * torch.arange(48000)),  # 1 s 440 Hz tone @ 48 kHz
                torch.sin(2 * 3.14159 * 1000 / 48000 * torch.arange(48000)),  # 1 s 1000 Hz tone @ 48 kHz
            ]),
            torch.stack([
                torch.sign(torch.sin(2 * 3.14159 * 200 / 48000 * torch.arange(48000))),  # 1 s 200 Hz square @ 48 kHz
                (1 + 2 * 200 / 48000 * torch.arange(48000)) % 2 - 1,  # 1 s 200 Hz sawtooth @ 48 kHz
            ]),
        ]),
        "fs": 48000,
        "reference": torch.tensor([
            [
                [1.1263639927, 2.1246092319, 3.6191856861, 1.2572505474, 1.8531025648],
                [1.2741736174, 1.8896869421, 3.3755991459, 1.2591584921, 1.6720581055],
            ],
            [
                [0.8731431961, 1.6447117329, 2.8125579357, 1.6197175980, 1.2627843618],
                [1.2543514967, 2.0644433498, 3.1744530201, 1.8767380714, 1.9447042942],
            ],
        ]),
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
    return out.mean(dim=0) if mean else out.reshape(*preds.shape[:-1], 5)


def _nisqa_cheat(preds, target, **kwargs: dict[str, Any]):
    # cheat the MetricTester as non_intrusive_speech_quality_assessment does not need a target
    return non_intrusive_speech_quality_assessment(preds, **kwargs)


class _NISQACheat(NonIntrusiveSpeechQualityAssessment):
    # cheat the MetricTester as NonIntrusiveSpeechQualityAssessment does not need a target
    def update(self, preds: Tensor, target: Tensor) -> None:
        super().update(preds=preds)


@pytest.mark.parametrize("preds, fs, reference", [(i["preds"], i["fs"], i["reference"]) for i in inputs])
class TestNISQA(MetricTester):
    """Test class for `NonIntrusiveSpeechQualityAssessment` metric."""

    atol = 1e-4

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_nisqa(self, preds: Tensor, reference: Tensor, fs: int, ddp: bool, device=None):
        """Test class implementation of metric."""
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
        self.run_functional_metric_test(
            preds=preds,
            target=preds,
            metric_functional=_nisqa_cheat,
            reference_metric=partial(_reference_metric_batch, mean=False),
            metric_args={"fs": fs},
        )


@pytest.mark.parametrize("shape", [(3000,), (2, 3000), (1, 2, 3000), (2, 3, 1, 3000)])
def test_shape(shape: tuple[int]):
    """Test output shape."""
    preds = torch.rand(*shape)
    out = non_intrusive_speech_quality_assessment(preds, 16000)
    assert out.shape == (*shape[:-1], 5)
    metric = NonIntrusiveSpeechQualityAssessment(16000)
    out = metric(preds)
    assert out.shape == (5,)


def test_batched_vs_unbatched():
    """Test batched versus unbatched processing."""
    preds = torch.rand(2, 2, 16000, generator=torch.Generator().manual_seed(42))
    out_batched = non_intrusive_speech_quality_assessment(preds, 16000)
    out_unbatched = torch.stack([
        non_intrusive_speech_quality_assessment(x, 16000) for x in preds.reshape(-1, 16000)
    ]).reshape(2, 2, 5)
    assert torch.allclose(out_batched, out_unbatched)


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
    with pytest.raises(RuntimeError, match="Maximum number of mel spectrogram windows exceeded. Use shorter audio."):
        non_intrusive_speech_quality_assessment(preds, 16000)
    non_intrusive_speech_quality_assessment(preds, 48000)
    preds = torch.rand(2502720)
    with pytest.raises(RuntimeError, match="Maximum number of mel spectrogram windows exceeded. Use shorter audio."):
        non_intrusive_speech_quality_assessment(preds, 16000)
    with pytest.raises(RuntimeError, match="Maximum number of mel spectrogram windows exceeded. Use shorter audio."):
        non_intrusive_speech_quality_assessment(preds, 48000)

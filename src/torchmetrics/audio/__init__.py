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
from torchmetrics.audio.pit import PermutationInvariantTraining
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
from torchmetrics.audio.snr import (
    ComplexScaleInvariantSignalNoiseRatio,
    ScaleInvariantSignalNoiseRatio,
    SignalNoiseRatio,
)
from torchmetrics.utilities.imports import (
    _GAMMATONE_AVAILABEL,
    _PESQ_AVAILABLE,
    _PYSTOI_AVAILABLE,
    _TORCHAUDIO_AVAILABEL,
    _TORCHAUDIO_GREATER_EQUAL_0_10,
)

__all__ = [
    "PermutationInvariantTraining",
    "ScaleInvariantSignalDistortionRatio",
    "SignalDistortionRatio",
    "ScaleInvariantSignalNoiseRatio",
    "SignalNoiseRatio",
    "ComplexScaleInvariantSignalNoiseRatio",
]

if _PESQ_AVAILABLE:
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

    __all__.append("PerceptualEvaluationSpeechQuality")

if _PYSTOI_AVAILABLE:
    from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

    __all__.append("ShortTimeObjectiveIntelligibility")

if _GAMMATONE_AVAILABEL and _TORCHAUDIO_AVAILABEL and _TORCHAUDIO_GREATER_EQUAL_0_10:
    from torchmetrics.audio.srmr import SpeechReverberationModulationEnergyRatio

    __all__.append("SpeechReverberationModulationEnergyRatio")

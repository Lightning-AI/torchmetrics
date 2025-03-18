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
from torchmetrics.audio.sdr import (
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
    SourceAggregatedSignalDistortionRatio,
)
from torchmetrics.audio.snr import (
    ComplexScaleInvariantSignalNoiseRatio,
    ScaleInvariantSignalNoiseRatio,
    SignalNoiseRatio,
)
from torchmetrics.utilities.imports import (
    _GAMMATONE_AVAILABLE,
    _LIBROSA_AVAILABLE,
    _ONNXRUNTIME_AVAILABLE,
    _PESQ_AVAILABLE,
    _PYSTOI_AVAILABLE,
    _REQUESTS_AVAILABLE,
    _SCIPI_AVAILABLE,
    _TORCHAUDIO_AVAILABLE,
)

if _SCIPI_AVAILABLE:
    import scipy.signal

    # back compatibility patch due to SMRMpy using scipy.signal.hamming
    if not hasattr(scipy.signal, "hamming"):
        scipy.signal.hamming = scipy.signal.windows.hamming

__all__ = [
    "ComplexScaleInvariantSignalNoiseRatio",
    "PermutationInvariantTraining",
    "ScaleInvariantSignalDistortionRatio",
    "ScaleInvariantSignalNoiseRatio",
    "SignalDistortionRatio",
    "SignalNoiseRatio",
    "SourceAggregatedSignalDistortionRatio",
]

if _PESQ_AVAILABLE:
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

    __all__ += ["PerceptualEvaluationSpeechQuality"]

if _PYSTOI_AVAILABLE:
    from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

    __all__ += ["ShortTimeObjectiveIntelligibility"]

if _GAMMATONE_AVAILABLE and _TORCHAUDIO_AVAILABLE:
    from torchmetrics.audio.srmr import SpeechReverberationModulationEnergyRatio

    __all__ += ["SpeechReverberationModulationEnergyRatio"]

if _LIBROSA_AVAILABLE and _ONNXRUNTIME_AVAILABLE:
    from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore

    __all__ += ["DeepNoiseSuppressionMeanOpinionScore"]

if _LIBROSA_AVAILABLE and _REQUESTS_AVAILABLE:
    from torchmetrics.audio.nisqa import NonIntrusiveSpeechQualityAssessment

    __all__ += ["NonIntrusiveSpeechQualityAssessment"]

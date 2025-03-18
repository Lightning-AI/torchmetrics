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
from torchmetrics.functional.audio.pit import permutation_invariant_training, pit_permutate
from torchmetrics.functional.audio.sdr import (
    scale_invariant_signal_distortion_ratio,
    signal_distortion_ratio,
    source_aggregated_signal_distortion_ratio,
)
from torchmetrics.functional.audio.snr import (
    complex_scale_invariant_signal_noise_ratio,
    scale_invariant_signal_noise_ratio,
    signal_noise_ratio,
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
    "complex_scale_invariant_signal_noise_ratio",
    "permutation_invariant_training",
    "pit_permutate",
    "scale_invariant_signal_distortion_ratio",
    "scale_invariant_signal_noise_ratio",
    "signal_distortion_ratio",
    "signal_noise_ratio",
    "source_aggregated_signal_distortion_ratio",
]

if _PESQ_AVAILABLE:
    from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality

    __all__ += ["perceptual_evaluation_speech_quality"]

if _PYSTOI_AVAILABLE:
    from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility

    __all__ += ["short_time_objective_intelligibility"]

if _GAMMATONE_AVAILABLE and _TORCHAUDIO_AVAILABLE:
    from torchmetrics.functional.audio.srmr import speech_reverberation_modulation_energy_ratio

    __all__ += ["speech_reverberation_modulation_energy_ratio"]

if _LIBROSA_AVAILABLE and _ONNXRUNTIME_AVAILABLE:
    from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

    __all__ += ["deep_noise_suppression_mean_opinion_score"]

if _LIBROSA_AVAILABLE and _REQUESTS_AVAILABLE:
    from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment

    __all__ += ["non_intrusive_speech_quality_assessment"]

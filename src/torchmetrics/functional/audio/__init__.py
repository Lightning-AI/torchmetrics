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
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio, signal_distortion_ratio
from torchmetrics.functional.audio.snr import (
    complex_scale_invariant_signal_noise_ratio,
    scale_invariant_signal_noise_ratio,
    signal_noise_ratio,
)
from torchmetrics.utilities.imports import (
    _GAMMATONE_AVAILABEL,
    _PESQ_AVAILABLE,
    _PYSTOI_AVAILABLE,
    _TORCHAUDIO_AVAILABEL,
    _TORCHAUDIO_GREATER_EQUAL_0_10,
)

__all__ = [
    "permutation_invariant_training",
    "pit_permutate",
    "scale_invariant_signal_distortion_ratio",
    "signal_distortion_ratio",
    "scale_invariant_signal_noise_ratio",
    "signal_noise_ratio",
    "complex_scale_invariant_signal_noise_ratio",
]

if _PESQ_AVAILABLE:
    from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality

    __all__.append("perceptual_evaluation_speech_quality")

if _PYSTOI_AVAILABLE:
    from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility

    __all__.append("short_time_objective_intelligibility")

if _GAMMATONE_AVAILABEL and _TORCHAUDIO_AVAILABEL and _TORCHAUDIO_GREATER_EQUAL_0_10:
    from torchmetrics.functional.audio.srmr import speech_reverberation_modulation_energy_ratio

    __all__.append("speech_reverberation_modulation_energy_ratio")

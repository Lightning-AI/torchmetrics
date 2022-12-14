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
from torchmetrics.functional.audio.pit import permutation_invariant_training, pit_permutate  # noqa: F401
from torchmetrics.functional.audio.sdr import (  # noqa: F401
    scale_invariant_signal_distortion_ratio,
    signal_distortion_ratio,
)
from torchmetrics.functional.audio.snr import scale_invariant_signal_noise_ratio, signal_noise_ratio  # noqa: F401
from torchmetrics.utilities.imports import _PESQ_AVAILABLE, _PYSTOI_AVAILABLE

if _PESQ_AVAILABLE:
    from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality  # noqa: F401

if _PYSTOI_AVAILABLE:
    from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility  # noqa: F401

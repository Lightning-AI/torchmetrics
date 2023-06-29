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
from torchmetrics.image.d_lambda import SpectralDistortionIndex
from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.psnrb import PeakSignalNoiseRatioWithBlockedEffect
from torchmetrics.image.rase import RelativeAverageSpectralError
from torchmetrics.image.rmse_sw import RootMeanSquaredErrorUsingSlidingWindow
from torchmetrics.image.sam import SpectralAngleMapper
from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure
from torchmetrics.image.tv import TotalVariation
from torchmetrics.image.uqi import UniversalImageQualityIndex
from torchmetrics.utilities.imports import _LPIPS_AVAILABLE, _TORCH_FIDELITY_AVAILABLE

__all__ = [
    "SpectralDistortionIndex",
    "ErrorRelativeGlobalDimensionlessSynthesis",
    "PeakSignalNoiseRatio",
    "PeakSignalNoiseRatioWithBlockedEffect",
    "RelativeAverageSpectralError",
    "RootMeanSquaredErrorUsingSlidingWindow",
    "SpectralAngleMapper",
    "MultiScaleStructuralSimilarityIndexMeasure",
    "MemorizationInformedFrechetInceptionDistance",
    "StructuralSimilarityIndexMeasure",
    "UniversalImageQualityIndex",
    "TotalVariation",
]

if _TORCH_FIDELITY_AVAILABLE:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.kid import KernelInceptionDistance

    __all__ += [
        "FrechetInceptionDistance",
        "InceptionScore",
        "KernelInceptionDistance",
    ]

if _LPIPS_AVAILABLE:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    __all__.append("LearnedPerceptualImagePatchSimilarity")

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
from torchmetrics.image.d_lambda import SpectralDistortionIndex  # noqa: F401
from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis  # noqa: F401
from torchmetrics.image.psnr import PeakSignalNoiseRatio  # noqa: F401
from torchmetrics.image.sam import SpectralAngleMapper  # noqa: F401
from torchmetrics.image.ssim import (  # noqa: F401
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.uqi import UniversalImageQualityIndex  # noqa: F401
from torchmetrics.utilities.imports import _LPIPS_AVAILABLE, _TORCH_FIDELITY_AVAILABLE

if _TORCH_FIDELITY_AVAILABLE:
    from torchmetrics.image.fid import FrechetInceptionDistance  # noqa: F401
    from torchmetrics.image.inception import InceptionScore  # noqa: F401
    from torchmetrics.image.kid import KernelInceptionDistance  # noqa: F401

if _LPIPS_AVAILABLE:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity  # noqa: F401

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
from torchmetrics.functional.image.d_lambda import spectral_distortion_index  # noqa: F401
from torchmetrics.functional.image.ergas import error_relative_global_dimensionless_synthesis  # noqa: F401
from torchmetrics.functional.image.gradients import image_gradients  # noqa: F401
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio  # noqa: F401
from torchmetrics.functional.image.sam import spectral_angle_mapper  # noqa: F401
from torchmetrics.functional.image.ssim import (  # noqa: F401
    multiscale_structural_similarity_index_measure,
    structural_similarity_index_measure,
)
from torchmetrics.functional.image.uqi import universal_image_quality_index  # noqa: F401

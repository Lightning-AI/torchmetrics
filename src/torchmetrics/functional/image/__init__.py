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
from torchmetrics.functional.image.d_lambda import spectral_distortion_index
from torchmetrics.functional.image.ergas import error_relative_global_dimensionless_synthesis
from torchmetrics.functional.image.gradients import image_gradients
from torchmetrics.functional.image.psnr import peak_signal_noise_ratio
from torchmetrics.functional.image.psnrb import peak_signal_noise_ratio_with_blocked_effect
from torchmetrics.functional.image.rase import relative_average_spectral_error
from torchmetrics.functional.image.rmse_sw import root_mean_squared_error_using_sliding_window
from torchmetrics.functional.image.sam import spectral_angle_mapper
from torchmetrics.functional.image.ssim import (
    multiscale_structural_similarity_index_measure,
    structural_similarity_index_measure,
)
from torchmetrics.functional.image.tv import total_variation
from torchmetrics.functional.image.uqi import universal_image_quality_index

__all__ = [
    "spectral_distortion_index",
    "error_relative_global_dimensionless_synthesis",
    "image_gradients",
    "peak_signal_noise_ratio",
    "peak_signal_noise_ratio_with_blocked_effect",
    "relative_average_spectral_error",
    "root_mean_squared_error_using_sliding_window",
    "spectral_angle_mapper",
    "multiscale_structural_similarity_index_measure",
    "structural_similarity_index_measure",
    "total_variation",
    "universal_image_quality_index",
]

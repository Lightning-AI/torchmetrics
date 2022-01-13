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
from typing import Optional, Sequence, Tuple

import torch
from deprecate import deprecated, void
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.image.ssim import multiscale_structural_similarity_index_measure as _ms_ssim


@deprecated(target=_ms_ssim, deprecated_in="0.7", remove_in="0.8")
def multiscale_structural_similarity_index_measure(
    preds: Tensor,
    target: Tensor,
    kernel_size: Sequence[int] = (11, 11),
    sigma: Sequence[float] = (1.5, 1.5),
    reduction: str = "elementwise_mean",
    data_range: Optional[float] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    betas: Tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
    normalize: Optional[Literal["relu", "simple"]] = None,
) -> Tensor:
    """Computes `MultiScaleSSIM`_, Multi-scale Structual Similarity Index Measure, which is a generalization of
    Structual Similarity Index Measure by incorporating image details at different resolution scores.

    Example:
        >>> preds = torch.rand([1, 1, 256, 256], generator=torch.manual_seed(42))
        >>> target = preds * 0.75
        >>> multiscale_structural_similarity_index_measure(preds, target)
        tensor(0.9558)
    """
    return void(preds, target, kernel_size, sigma, reduction, data_range, k1, k2, betas, normalize)

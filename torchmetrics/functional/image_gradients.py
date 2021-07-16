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
from typing import Tuple
from warnings import warn

import torch
from torch import Tensor

from torchmetrics.functional.image.image_gradients import image_gradients as _image_gradients


def image_gradients(img: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Computes the `gradients <https://en.wikipedia.org/wiki/Image_gradient>`_ of a given image using finite difference

    .. deprecated:: v0.5
        Use :func:`torchmetrics.functional.image.image_gradients.image_gradients`. Will be removed in v0.6.

    """
    warn(
        "Function `functional.image_gradients.image_gradients` is deprecated in v0.5 and will be removed in v0.6."
        " Use `functional.image.image_gradients.image_gradients` instead.", DeprecationWarning
    )

    return _image_gradients(img)

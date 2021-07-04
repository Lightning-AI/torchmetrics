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
from typing import Any, Optional, Tuple, Union
from warnings import warn

from torchmetrics.image.psnr import PSNR as _PSNR


class PSNR(_PSNR):
    """
    .. deprecated:: v0.4
        The PSNR was moved to `torchmetrics.image.psnr`. It will be removed in v0.5.

    """

    def __init__(
        self,
        data_range: Optional[float] = None,
        base: float = 10.0,
        reduction: str = 'elementwise_mean',
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        warn(
            "This `PSNR` was moved in v0.4 and this shell will be removed in v0.5."
            " Use `torchmetrics.image.psnr` instead.", DeprecationWarning
        )
        super().__init__(
            data_range=data_range,
            base=base,
            reduction=reduction,
            dim=dim,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

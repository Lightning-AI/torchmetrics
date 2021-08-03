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
from typing import Any, Callable, Optional
from warnings import warn

from torchmetrics.regression.r2 import R2Score as _R2Score


class R2Score(_R2Score):
    r"""
    Computes r2 score also known as `coefficient of determination
    <https://en.wikipedia.org/wiki/Coefficient_of_determination>`_:

    .. deprecated:: v0.5
        `r2score` was renamed as `r2_score` in v0.5 and it will be removed in v0.6
    """

    def __init__(
        self,
        num_outputs: int = 1,
        adjusted: int = 0,
        multioutput: str = "uniform_average",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        warn(
            "`R2Score` was moved from `torchmetrics.regression.r2` to `torchmetrics.regression.r2_score` in v0.5"
            " and it will be removed in v0.6",
            DeprecationWarning,
        )
        super().__init__(
            num_outputs=num_outputs,
            adjusted=adjusted,
            multioutput=multioutput,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

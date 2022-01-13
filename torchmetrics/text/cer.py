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

from typing import Any, Callable, List, Optional, Union

import torch
from deprecate import deprecated
from torch import Tensor, tensor

from torchmetrics.functional.text.cer import _cer_compute, _cer_update
from torchmetrics.metric import Metric


class CharErrorRate(Metric):
    r"""
    Character error rate (CharErrorRate_) is a metric of the performance of an automatic speech recognition
    (ASR) system. This value indicates the percentage of characters that were incorrectly predicted.
    The lower the value, the better the performance of the ASR system with a CharErrorRate of 0 being
    a perfect score.
    Character error rate can then be computed as:

    .. math::
        CharErrorRate = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}

    where:
        - S is the number of substitutions,
        - D is the number of deletions,
        - I is the number of insertions,
        - C is the number of correct characters,
        - N is the number of characters in the reference (N=S+D+C).

    Compute CharErrorRate score of transcribed segments against references.

    Args:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called.
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    Returns:
        Character error rate score

    Examples:
        >>> preds = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> metric = CharErrorRate()
        >>> metric(preds, target)
        tensor(0.3415)
    """
    is_differentiable = False
    higher_is_better = False
    error: Tensor
    total: Tensor

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("errors", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    @deprecated(
        args_mapping={"predictions": "preds", "references": "target"},
        target=True,
        deprecated_in="0.7",
        remove_in="0.8",
    )
    def update(self, preds: Union[str, List[str]], target: Union[str, List[str]]) -> None:  # type: ignore
        """Store references/predictions for computing Character Error Rate scores.

        Args:
            preds: Transcription(s) to score as a string or list of strings
            target: Reference(s) for each speech input as a string or list of strings

        .. deprecated:: v0.7
            Args:
                predictions:
                    This argument is deprecated in favor of  `preds` and will be removed in v0.8.
                references:
                    This argument is deprecated in favor of  `target` and will be removed in v0.8.
        """
        errors, total = _cer_update(preds, target)
        self.errors += errors
        self.total += total

    def compute(self) -> Tensor:
        """Calculate the character error rate.

        Returns:
           Character error rate score
        """
        return _cer_compute(self.errors, self.total)

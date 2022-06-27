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

from typing import Any, List, Union

import torch
from torch import Tensor, tensor

from torchmetrics.functional.text.cer import _cer_compute, _cer_update
from torchmetrics.metric import Metric


class CharErrorRate(Metric):
    r"""Character Error Rate (CER_) is a metric of the performance of an automatic speech recognition
    (ASR) system.

    This value indicates the percentage of characters that were incorrectly predicted.
    The lower the value, the better the performance of the ASR system with a CharErrorRate of 0 being
    a perfect score.
    Character error rate can then be computed as:

    .. math::
        CharErrorRate = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}

    where:
        - :math:`S` is the number of substitutions,
        - :math:`D` is the number of deletions,
        - :math:`I` is the number of insertions,
        - :math:`C` is the number of correct characters,
        - :math:`N` is the number of characters in the reference (N=S+D+C).

    Compute CharErrorRate score of transcribed segments against references.

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        Character error rate score

    Examples:
        >>> preds = ["this is the prediction", "there is an other sample"]
        >>> target = ["this is the reference", "there is another one"]
        >>> metric = CharErrorRate()
        >>> metric(preds, target)
        tensor(0.3415)
    """
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    error: Tensor
    total: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.add_state("errors", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: Union[str, List[str]], target: Union[str, List[str]]) -> None:  # type: ignore
        """Store references/predictions for computing Character Error Rate scores.

        Args:
            preds: Transcription(s) to score as a string or list of strings
            target: Reference(s) for each speech input as a string or list of strings
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

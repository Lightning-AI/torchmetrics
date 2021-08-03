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

from torchmetrics.functional import wer
from torchmetrics.metric import Metric


class WER(Metric):
    r"""
    Word error rate (WER_) is a common metric of
    the performance of an automatic speech recognition system.
    This value indicates the percentage of words that were incorrectly predicted.
    The lower the value, the better the performance of the ASR system with a WER of 0 being a perfect score.
    Word error rate can then be computed as:

    .. math::
        WER = \frac{S + D + I}{N} = \frac{S + D + I}{S + D + C}

    where:

        - S is the number of substitutions,
        - D is the number of deletions,
        - I is the number of insertions,
        - C is the number of correct words,
        - N is the number of words in the reference (N=S+D+C).

    Compute WER score of transcribed segments against references.

    Args:
        concatenate_texts: Whether to concatenate all input texts or compute WER iteratively.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    Returns:
        (float): the word error rate

    Examples:
        >>> predictions = ["this is the prediction", "there is an other sample"]
        >>> references = ["this is the reference", "there is another one"]
        >>> metric = WER()
        >>> metric(predictions, references)
        0.5

    """

    def __init__(
        self,
        concatenate_texts: bool = False,
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
        self.concatenate_texts = concatenate_texts
        self.add_state("predictions", [], dist_reduce_fx="cat")
        self.add_state("references", [], dist_reduce_fx="cat")

    def update(self, predictions: Union[str, List[str]], references: Union[str, List[str]]) -> None:  # type: ignore
        """Store predictions/references for computing Word Error Rate scores.

        Args:
            predictions: List of transcriptions to score.
            references: List of references for each speech input.
        """
        self.predictions.append(predictions)
        self.references.append(references)

    def compute(self) -> float:
        """Calculate Word Error Rate scores.

        Return:
            Float with WER Score.
        """
        return wer(self.references, self.predictions, concatenate_texts=self.concatenate_texts)

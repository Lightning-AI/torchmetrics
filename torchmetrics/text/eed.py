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

from typing_extensions import Literal
from typing import Any, Callable, Optional, Sequence, Union

from torch import Tensor, tensor

from torchmetrics.functional.text.eed import _eed_compute, _eed_update
from torchmetrics.metric import Metric


class EED(Metric):
    """Computes extended edit distance score (`EED`_) [1] for strings or list of strings The metric utilises the
    Levenshtein distance and extends it by adding an additional jump operation.

    Args:
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now. Defaults to en

    Returns:
        Extended edit distance score as a tensor

    Example:
        >>> hypotheses = ["this is the prediction", "here is an other sample"]
        >>> references = ["this is the reference", "here is another one"]
        >>> metric = EED()
        >>> metric(hypotheses=hypotheses, references=references)
        tensor(0.3078)

    References:
        [1] P. Stanchev, W. Wang, and H. Ney, “EED: Extended Edit Distance Measure for Machine Translation”, submitted
        to WMT 2019. `EED`_
    """

    scores: Tensor
    total: Tensor

    def __init__(
        self,
        language: Literal["en", "ja"] = "en",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__()
        self.language: Literal["en", "ja"] = language
        self.add_state("scores", tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        reference_corpus: Sequence[Union[str, Sequence[str]]],
        hypothesis_corpus: Union[str, Sequence[str]],
    ) -> None:
        """Update EED statistics.

        Args:
            hypotheses: Transcription(s) to score as a string or list of strings
            references: Reference(s) for each input as a string or list of strings

        Returns:
            None
        """
        scores, total = _eed_update(hypotheses=reference_corpus, references=hypothesis_corpus, language=self.language)
        self.scores += scores
        self.total += total

    def compute(self) -> Tensor:
        """Calculate extended edit distance score.

        Returns:
            Extended edit distance score as tensor
        """
        eed = _eed_compute(self.scores, self.total)
        return eed

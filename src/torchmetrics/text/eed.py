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

from typing import Any, List, Sequence, Tuple, Union

from torch import Tensor, stack
from typing_extensions import Literal

from torchmetrics.functional.text.eed import _eed_compute, _eed_update
from torchmetrics.metric import Metric


class ExtendedEditDistance(Metric):
    """Computes extended edit distance score (`ExtendedEditDistance`_) [1] for strings or list of strings.

    The metric utilises the Levenshtein distance and extends it by adding a jump operation.

    Args:
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now.
        return_sentence_level_score: An indication of whether sentence-level EED score is to be returned
        alpha: optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Return:
        Extended edit distance score as a tensor

    Example:
        >>> from torchmetrics import ExtendedEditDistance
        >>> preds = ["this is the prediction", "here is an other sample"]
        >>> target = ["this is the reference", "here is another one"]
        >>> metric = ExtendedEditDistance()
        >>> metric(preds=preds, target=target)
        tensor(0.3078)

    References:
        [1] P. Stanchev, W. Wang, and H. Ney, “EED: Extended Edit Distance Measure for Machine Translation”, submitted
        to WMT 2019. `ExtendedEditDistance`_
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    sentence_eed: List[Tensor]

    def __init__(
        self,
        language: Literal["en", "ja"] = "en",
        return_sentence_level_score: bool = False,
        alpha: float = 2.0,
        rho: float = 0.3,
        deletion: float = 0.2,
        insertion: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if language not in ("en", "ja"):
            raise ValueError(f"Expected argument `language` to either be `en` or `ja` but got {language}")
        self.language: Literal["en", "ja"] = language
        self.return_sentence_level_score = return_sentence_level_score

        # input validation for parameters
        for param_name, param in zip(["alpha", "rho", "deletion", "insertion"], [alpha, rho, deletion, insertion]):
            if not isinstance(param, float) or isinstance(param, float) and param < 0:
                raise ValueError(f"Parameter `{param_name}` is expected to be a non-negative float.")

        self.alpha = alpha
        self.rho = rho
        self.deletion = deletion
        self.insertion = insertion

        self.add_state("sentence_eed", [], dist_reduce_fx="cat")

    def update(  # type: ignore
        self,
        preds: Union[str, Sequence[str]],
        target: Sequence[Union[str, Sequence[str]]],
    ) -> None:
        """Update ExtendedEditDistance statistics.

        Args:
            preds: An iterable of hypothesis corpus
            target: An iterable of iterables of reference corpus
        """
        self.sentence_eed = _eed_update(
            preds,
            target,
            self.language,
            self.alpha,
            self.rho,
            self.deletion,
            self.insertion,
            self.sentence_eed,
        )

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Calculate extended edit distance score.

        Return:
            Extended edit distance score as tensor
        """
        average = _eed_compute(self.sentence_eed)

        if self.return_sentence_level_score:
            return average, stack(self.sentence_eed)
        return average

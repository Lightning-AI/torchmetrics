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
from typing import Any, List, Optional, Sequence, Tuple, Union

from torch import Tensor, stack
from typing_extensions import Literal

from torchmetrics.functional.text.eed import _eed_compute, _eed_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["ExtendedEditDistance.plot"]


class ExtendedEditDistance(Metric):
    """Compute extended edit distance score (`ExtendedEditDistance`_) for strings or list of strings.

    The metric utilises the Levenshtein distance and extends it by adding a jump operation.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~Sequence`): An iterable of hypothesis corpus
    - ``target`` (:class:`~Sequence`): An iterable of iterables of reference corpus

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``eed`` (:class:`~torch.Tensor`): A tensor with the extended edit distance score

    Args:
        language: Language used in sentences. Only supports English (en) and Japanese (ja) for now.
        return_sentence_level_score: An indication of whether sentence-level EED score is to be returned
        alpha: optimal jump penalty, penalty for jumps between characters
        rho: coverage cost, penalty for repetition of characters
        deletion: penalty for deletion of character
        insertion: penalty for insertion or substitution of character
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.text import ExtendedEditDistance
        >>> preds = ["this is the prediction", "here is an other sample"]
        >>> target = ["this is the reference", "here is another one"]
        >>> eed = ExtendedEditDistance()
        >>> eed(preds=preds, target=target)
        tensor(0.3078)
    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

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
    ) -> None:
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

    def update(
        self,
        preds: Union[str, Sequence[str]],
        target: Sequence[Union[str, Sequence[str]]],
    ) -> None:
        """Update state with predictions and targets."""
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
        """Calculate extended edit distance score."""
        average = _eed_compute(self.sentence_eed)

        if self.return_sentence_level_score:
            return average, stack(self.sentence_eed)
        return average

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> from torchmetrics.text import ExtendedEditDistance
            >>> metric = ExtendedEditDistance()
            >>> preds = ["this is the prediction", "there is an other sample"]
            >>> target = ["this is the reference", "there is another one"]
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.text import ExtendedEditDistance
            >>> metric = ExtendedEditDistance()
            >>> preds = ["this is the prediction", "there is an other sample"]
            >>> target = ["this is the reference", "there is another one"]
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)

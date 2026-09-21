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

from collections.abc import Sequence
from typing import Any, List, Optional, Union

import torch
from torch import Tensor, tensor

from torchmetrics.functional.text.meteor import _ensure_nltk_wordnet_is_downloaded, _meteor_score_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _NLTK_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["METEORScore.plot"]

if not _NLTK_AVAILABLE:
    __doctest_skip__ = ["METEORScore", "METEORScore.plot"]


class METEORScore(Metric):
    """Calculate `METEOR`_ score of machine translated text with one or more references.

    METEOR aligns the hypothesis to each reference using exact, stemmed and WordNet synonym matches, scores the
    alignment with a recall weighted harmonic mean of precision and recall, and applies a fragmentation penalty. When
    several references are given the best scoring reference is used, and the corpus level score is the average over all
    hypotheses.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~Sequence`): An iterable of hypothesis corpus
    - ``target`` (:class:`~Sequence`): An iterable of iterables of reference corpus

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``meteor`` (:class:`~torch.Tensor`): if ``return_sentence_level_score=True`` return a corpus level METEOR score
      together with a list of sentence level scores, else return a corpus level METEOR score

    Args:
        alpha: Parameter controlling the relative weight of precision and recall.
        beta: Parameter controlling the shape of the fragmentation penalty.
        gamma: Relative weight assigned to the fragmentation penalty.
        return_sentence_level_score: An indication whether a sentence level METEOR score to be returned.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ModuleNotFoundError:
            If ``nltk`` package is not installed.

    Example:
        >>> from torchmetrics.text import METEORScore
        >>> preds = ['the cat sat on the mat']
        >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
        >>> meteor = METEORScore()
        >>> meteor(preds, target)
        tensor(0.6250)

    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    score: Tensor
    total: Tensor
    sentence_meteor: Optional[List[Tensor]] = None

    def __init__(
        self,
        alpha: float = 0.9,
        beta: float = 3.0,
        gamma: float = 0.5,
        return_sentence_level_score: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not _NLTK_AVAILABLE:
            raise ModuleNotFoundError("METEOR metric requires that `nltk` is installed. Use `pip install nltk`.")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Expected argument `alpha` to be in the [0, 1] range but got {alpha}.")
        if beta < 0.0:
            raise ValueError(f"Expected argument `beta` to be non-negative but got {beta}.")
        if gamma < 0.0:
            raise ValueError(f"Expected argument `gamma` to be non-negative but got {gamma}.")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.return_sentence_level_score = return_sentence_level_score

        self.add_state("score", tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", tensor(0.0), dist_reduce_fx="sum")
        if self.return_sentence_level_score:
            self.add_state("sentence_meteor", [], dist_reduce_fx="cat")

    def update(self, preds: Union[str, Sequence[str]], target: Union[Sequence[str], Sequence[Sequence[str]]]) -> None:
        """Update state with predictions and targets."""
        import nltk

        _ensure_nltk_wordnet_is_downloaded()
        stemmer = nltk.stem.porter.PorterStemmer()

        results = _meteor_score_update(preds, target, stemmer, nltk.corpus.wordnet, self.alpha, self.beta, self.gamma)
        for result in results:
            self.score += result
            self.total += 1
        if self.sentence_meteor is not None:
            self.sentence_meteor.extend(result.unsqueeze(0) for result in results)

    def compute(self) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Calculate the METEOR score."""
        meteor = self.score / self.total if self.total > 0 else tensor(0.0)
        if self.sentence_meteor is not None:
            return meteor, torch.cat(self.sentence_meteor)
        return meteor

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
            >>> from torchmetrics.text import METEORScore
            >>> metric = METEORScore()
            >>> preds = ['the cat sat on the mat']
            >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.text import METEORScore
            >>> metric = METEORScore()
            >>> preds = ['the cat sat on the mat']
            >>> target = [['there is a cat on the mat', 'a cat is on the mat']]
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)

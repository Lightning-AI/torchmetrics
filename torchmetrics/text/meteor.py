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
# referenced from
# Library Name: torchtext
# Authors: torchtext authors
# Date: 2021-11-15

import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics import Metric
from torchmetrics.functional.text.meteor import (
    _meteor_score_compute,
    _meteor_score_update,
    _METEORScoreComponents,
    _NLTKStemmerWrapper,
    _NLTKWordnetWrapper,
)
from torchmetrics.utilities.imports import _NLTK_AVAILABLE


class METEORScore(Metric):
    """Calculate the [METEOR](https://en.wikipedia.org/wiki/METEOR) (Metric for Evaluation of Translation with
    Explicit ORdering) score of machine translated text with one or more references. This metric was designed to
    fix some of the problems found in the more popular BLEU metric, and also produce good correlation with human
    judgement at the sentence or segment level.

    Args:
        stemmer:
            A name of stemmer from `nltk` package to be used.
        wordnet:
            A name of wordnet corpus from `nltk` package to be used.
        alpha:
            A parameter for controlling relative weights of precision and recall.
        beta:
            A parameter for controlling shape of penalty as a function of as a function of fragmentation.
        gamma:
            A relative weight assigned to fragmentation penalty.
        compute_on_step:
            Forward only calls ``update()`` and returns None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather.

    Example:
        >>> import nltk
        >>> nltk.download('wordnet')
        True
        >>> predictions = ['the cat is on the mat']
        >>> references = [['there is a cat on the mat']]
        >>> meteor_score = METEORScore()
        >>> meteor_score.update(references, predictions)
        >>> meteor_score.compute()
        tensor([0.6464])

    References:
    [1] METEOR: An Automatic Metric for MT Evaluation with High Levels of Correlation with Human Judgments by Alon
    Lavie and Abhaya Agarwal.
    """

    is_differentiable = False
    higher_is_better = True

    def __init__(
        self,
        stemmer: Literal["porter"] = "porter",
        wordnet: Literal["wordnet"] = "wordnet",
        alpha: float = 0.9,
        beta: float = 3.0,
        gamma: float = 0.5,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ):
        warnings.warn(
            "Current implementation follows the original METEOR metric and thus is not suitable for reporting results "
            "in research papers."
        )
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if not _NLTK_AVAILABLE:
            raise ValueError(
                "METEORScore metric requires that `nltk` is installed. "
                "Either install with `pip install nltk` or `pip install torchmetrics[text]`"
            )

        self.stemmer = _NLTKStemmerWrapper(stemmer)
        self.wordnet = _NLTKWordnetWrapper(wordnet)

        if not 0 <= alpha <= 1:
            raise ValueError("Expected `alpha` argument to be between 0 and 1")
        self.alpha = alpha
        self.beta = beta
        if not 0 <= gamma <= 1:
            raise ValueError("Expected `gamma` argument to be between 0 and 1")
        self.gamma = gamma

        self.meteor_score_components: List[Tuple[_METEORScoreComponents, ...]] = []

    def update(  # type: ignore
        self,
        reference_corpus: Union[List[str], List[List[str]]],
        hypothesis_corpus: Union[str, List[str]],
    ) -> None:
        """Updates state with METEORScore calculated on the input.

        Args:
            reference_corpus: Either an list of reference corpus or an list of lists of reference corpus
            hypothesis_corpus: Either an single reference corpus or an list of reference corpus
        """
        if isinstance(hypothesis_corpus, str):
            hypothesis_corpus = [hypothesis_corpus]

        if len(reference_corpus) > 0 and isinstance(reference_corpus[0], str):
            if len(hypothesis_corpus) == 1:
                reference_corpus = [reference_corpus]
            else:
                reference_corpus = [[reference] for reference in reference_corpus]

        if len(reference_corpus) != len(hypothesis_corpus):
            raise ValueError(f"Corpus has different size {len(reference_corpus)} != {len(hypothesis_corpus)}")
        print(hypothesis_corpus)
        self.meteor_score_components.extend(
            _meteor_score_update(reference_corpus, hypothesis_corpus, self.stemmer, self.wordnet)
        )
        print(self.meteor_score_components)

    def compute(self) -> Tensor:
        """Calculate METEOR score.

        Return:
            Tensor with METEOR Score
        """
        return _meteor_score_compute(self.meteor_score_components, self.alpha, self.beta, self.gamma)

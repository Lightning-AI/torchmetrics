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

from typing import Any, Callable, List, Optional, Tuple, Union

from torch import Tensor

from torchmetrics import Metric
from torchmetrics.functional.text.meteor import (
    _SUPPORTED_STEMMER_LANGUAGES_TYPE,
    _get_function_words,
    _meteor_score_compute,
    _meteor_score_update,
    _METEORScoreComponents,
    _NLTKStemmerWrapper,
    _NLTKWordnetWrapper,
)
from torchmetrics.utilities.imports import _NLTK_AVAILABLE, _NLTK_CORPUS_AVAILABLE


def _meteor_dist_reduce_fx(x: List[Tuple[_METEORScoreComponents, ...]]) -> List[Tuple[_METEORScoreComponents, ...]]:
    """During DDP, individual components of `_METEORScoreComponents` instances contain the list of tensors.

    We need to unroll these lists to multiple `_METEORScoreComponents` instances.
    """

    def unroll_meteor_score_components(ms: _METEORScoreComponents) -> List[List[_METEORScoreComponents]]:
        ms_unrolled = [
            [_METEORScoreComponents(m, r, h, f)]
            for m, r, h, f in zip(ms.matches_count, ms.reference_len, ms.hypothesis_len, ms.frag_frac)
        ]
        return ms_unrolled

    for i in range(len(x)):
        ms_unrolled_list: List[List[_METEORScoreComponents]] = []
        for j in range(len(x[i])):
            if not ms_unrolled_list:
                ms_unrolled_list += unroll_meteor_score_components(x[i][j])
            else:
                ms_unrolled = unroll_meteor_score_components(x[i][j])
                for k in range(len(ms_unrolled_list)):
                    ms_unrolled_list[k] += ms_unrolled[k]

    return [tuple(y) for y in ms_unrolled_list]


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
            Expected `alpha` to be between 0 and 1.
        beta:
            A parameter for controlling shape of penalty as a function of as a function of fragmentation.
            Expected `beta` be greater than or equal to 0.
        gamma:
            A relative weight assigned to fragmentation penalty.
            Expected `gamma` to be between 0 and 1.
        delta:
            A relative weight to content and function words. Relevant only if `use_function_words = True`.
            Expected `delta` to be between 0 and 1.
        use_function_words:
            An indication whether to discriminate between content and function words. A list of function words is
            derived on monolingual corpora. All words with relative frequency above 10^{-3} are considered to be
            function words.
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

    Raises:
        ValueError:
            If `nltk` package is not installed.
        ValueError:
            If `alpha` is not between 0 and 1.
        ValueError:
            If `beta` is not greater than or equal to 0.
        ValueError:
            If `gamma` is not between 0 and 1.

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
    # meteor_score_components: List[Tuple[_METEORScoreComponents, ...]]. Definition below used for torch.jit.script
    # compatibility
    meteor_score_components: List[Any]

    def __init__(
        self,
        lang: _SUPPORTED_STEMMER_LANGUAGES_TYPE = "porter",
        alpha: float = 0.9,
        beta: float = 3.0,
        gamma: float = 0.5,
        delta: float = 0.75,
        use_function_words: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ):
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
        if not _NLTK_CORPUS_AVAILABLE:
            raise ValueError(
                "METEOR metric requires that nltk wordnet corpus is downloaded. "
                "Use `python -m nltk.downloader wordnet`."
            )

        self.stemmer = _NLTKStemmerWrapper(lang)
        self.wordnet = _NLTKWordnetWrapper()

        if not 0 <= alpha <= 1:
            raise ValueError("Expected `alpha` argument to be between 0 and 1")
        self.alpha = alpha
        if beta < 0:
            raise ValueError("Expected `beta` argument to be greater than or equal to 0.")
        self.beta = beta
        if not 0 <= gamma <= 1:
            raise ValueError("Expected `gamma` argument to be between 0 and 1")
        self.gamma = gamma
        if not 0 <= delta <= 1:
            raise ValueError("Expected `delta` argument to be between 0 and 1")
        self.delta = delta

        if use_function_words:
            self.function_words = _get_function_words(lang)
        else:
            self.function_words = ()

        self.add_state("meteor_score_components", [], dist_reduce_fx=_meteor_dist_reduce_fx)

    def update(  # type: ignore
        self,
        reference_corpus: Union[List[str], List[List[str]]],
        hypothesis_corpus: Union[str, List[str]],
    ) -> None:
        """Updates state with METEORScore calculated on the input.

        Args:
            reference_corpus:
                An iterable of iterables of reference corpus. Either a list of reference corpora or a list of lists of
                reference corpora.
            hypothesis_corpus:
                An iterable of machine translated corpus. Either a single hypothesis corpus or a list of hypothesis
                 corpora.
        """
        self.meteor_score_components.extend(
            _meteor_score_update(reference_corpus, hypothesis_corpus, self.stemmer, self.wordnet, self.function_words)
        )

    def compute(self) -> Tensor:
        """Calculate METEOR score.

        Return:
            Tensor with sentence-level METEOR Score
        """
        return _meteor_score_compute(self.meteor_score_components, self.alpha, self.beta, self.gamma, self.delta)

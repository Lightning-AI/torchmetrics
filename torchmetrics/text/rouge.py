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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch import Tensor

from torchmetrics import Metric
from torchmetrics.functional.text.rouge import ALLOWED_ROUGE_KEYS, _rouge_score_compute, _rouge_score_update
from torchmetrics.utilities.imports import _NLTK_AVAILABLE, _ROUGE_SCORE_AVAILABLE

if _ROUGE_SCORE_AVAILABLE:
    from rouge_score.rouge_scorer import RougeScorer
    from rouge_score.scoring import BootstrapAggregator
else:
    RougeScorer, BootstrapAggregator = object, object


class ROUGEScore(Metric):
    """Calculate `ROUGE score <https://en.wikipedia.org/wiki/ROUGE_(metric)>`_, used for automatic summarization.

    Args:
        newline_sep:
            New line separate the inputs.
        use_stemmer:
            Use Porter stemmer to strip word suffixes to improve matching.
        rouge_keys:
            A list of rouge types to calculate.
            Keys that are allowed are ``rougeL``, ``rougeLsum``, and ``rouge1`` through ``rouge9``.
        decimal_places:
            The number of digits to round the computed the values to.
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

        >>> targets = "Is your name John".split()
        >>> preds = "My name is John".split()
        >>> rouge = ROUGEScore()   # doctest: +SKIP
        >>> from pprint import pprint
        >>> pprint(rouge(preds, targets))  # doctest: +NORMALIZE_WHITESPACE +SKIP
        {'rouge1_fmeasure': 0.25,
         'rouge1_precision': 0.25,
         'rouge1_recall': 0.25,
         'rouge2_fmeasure': 0.0,
         'rouge2_precision': 0.0,
         'rouge2_recall': 0.0,
         'rougeL_fmeasure': 0.25,
         'rougeL_precision': 0.25,
         'rougeL_recall': 0.25,
         'rougeLsum_fmeasure': 0.25,
         'rougeLsum_precision': 0.25,
         'rougeLsum_recall': 0.25}

    Raises:
        ValueError:
            If the python packages ``nltk`` or ``rouge-score`` are not installed.
        ValueError:
            If any of the ``rouge_keys`` does not belong to the allowed set of keys.

    References:
        [1] ROUGE: A Package for Automatic Evaluation of Summaries by Chin-Yew Lin https://aclanthology.org/W04-1013/
    """

    def __init__(
        self,
        newline_sep: bool = False,
        use_stemmer: bool = False,
        rouge_keys: Union[str, Tuple[str, ...]] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),  # type: ignore
        decimal_places: int = 4,
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

        if not (_NLTK_AVAILABLE and _ROUGE_SCORE_AVAILABLE):
            raise ValueError(
                "ROUGE metric requires that both nltk and rouge-score is installed."
                " Either as `pip install torchmetrics[text]` or `pip install nltk rouge-score`"
            )

        if not isinstance(rouge_keys, tuple):
            rouge_keys = tuple([rouge_keys])
        for key in rouge_keys:
            if key not in ALLOWED_ROUGE_KEYS:
                raise ValueError(f"Got unknown rouge key {key}. Expected to be one of {ALLOWED_ROUGE_KEYS}")

        self.rouge_keys = rouge_keys
        self.newline_sep = newline_sep
        self.use_stemmer = use_stemmer
        self.aggregator = BootstrapAggregator()
        self.scorer = RougeScorer(rouge_keys, use_stemmer=self.use_stemmer)
        self.decimal_places = decimal_places

    def update(self, preds: Union[str, List[str]], targets: Union[str, List[str]]) -> None:  # type: ignore
        """Compute rouge scores.

        Args:
            preds: An iterable of predicted sentences.
            targets: An iterable of target sentences.
        """

        if isinstance(preds, str):
            preds = [preds]

        if isinstance(targets, str):
            targets = [targets]

        _rouge_score_update(
            preds, targets, scorer=self.scorer, aggregator=self.aggregator, newline_sep=self.newline_sep
        )

    def compute(self) -> Dict[str, Tensor]:
        """Calculate (Aggregate and provide confidence intervals) ROUGE score.

        Return:
            Python dictionary of rouge scores for each input rouge key.
        """
        return _rouge_score_compute(aggregator=self.aggregator, decimal_places=self.decimal_places)

    def __hash__(self) -> int:
        # override to hash list objects.
        # this is a bug in the upstream pytorch release.
        hash_vals = [self.__class__.__name__]

        for key in self._defaults:
            value = getattr(self, key)
            if isinstance(value, list):
                value = tuple(value)
            hash_vals.append(value)

        return hash(tuple(hash_vals))

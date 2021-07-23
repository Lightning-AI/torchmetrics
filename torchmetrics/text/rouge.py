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
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import tensor

from torchmetrics import Metric
from torchmetrics.functional.text.rouge import RougeBatchAggregator, _rouge_score_compute, _rouge_score_update
from torchmetrics.utilities.imports import _ROUGE_SCORE_AVAILABLE

if _ROUGE_SCORE_AVAILABLE:
    from rouge_score import rouge_scorer


class RougeMetric(Metric):
    """
    Calculate `ROUGE score <https://en.wikipedia.org/wiki/ROUGE_(metric)>`_.
    Used for automatic summarization.

    Args:
        pred_lns:
            An iterable of
        tgt_lns:
            An iterable of
        rouge_newline_sep:

        use_stemmer:

        rouge_keys

    Example:

        >>> target = "Is your name John".split()
        >>> preds = "My name is John".split()
        >>> rouge = RougeMetric()   # doctest: +SKIP
        >>> from pprint import pprint
        >>> pprint(rouge(preds, target))  # doctest: +NORMALIZE_WHITESPACE +SKIP
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

    References:
        [1] ROUGE: A Package for Automatic Evaluation of Summaries by Chin-Yew Lin https://aclanthology.org/W04-1013/
    """

    def __init__(
        self,
        rouge_newline_sep: bool = False,
        use_stemmer: bool = False,
        rouge_keys: Tuple[str] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),
    ):
        super().__init__()

        self.rouge_newline_sep = rouge_newline_sep
        self.rouge_keys = rouge_keys
        self.use_stemmer = use_stemmer
        self.aggregator = RougeBatchAggregator()
        self.scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=self.use_stemmer)
        self.scores = {key: [] for key in rouge_keys}

    def update(self, pred_lns: List[str], tgt_lns: List[str]) -> None:
        """
        Compute rouge scores.
        Args:
            pred_lns: An iterable of iterables of reference corpus
            tgt_lns: An iterable of machine translated corpus
        """
        _rouge_score_update(
            pred_lns, tgt_lns, scores=self.scores, scorer=self.scorer, rouge_newline_sep=self.rouge_newline_sep
        )

    def compute(self) -> Dict[str, float]:
        """
        Calculate (Agregate and provide confidence intervals) ROUGE score

        Return:
            Python dictionary of rouge scores for each input rouge key.
        """
        _rouge_score_compute(self.scores, aggregator=self.aggregator)

    def __hash__(self):
        # override to hash list objects.
        # this is a bug in the upstream pytorch release.
        hash_vals = [self.__class__.__name__]

        for key in self._defaults:
            value = getattr(self, key)
            if isinstance(value, list):
                value = tuple(value)
            hash_vals.append(value)

        return hash(tuple(hash_vals))

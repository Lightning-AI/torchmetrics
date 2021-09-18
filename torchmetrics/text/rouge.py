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
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch import Tensor

from torchmetrics import Metric
from torchmetrics.functional.text.rouge import ALLOWED_ROUGE_KEYS, _rouge_score_compute, _rouge_score_update
from torchmetrics.utilities.imports import _NLTK_AVAILABLE


class ROUGEScore(Metric):
    """Calculate `ROUGE score <https://en.wikipedia.org/wiki/ROUGE_(metric)>`_, used for automatic summarization.
    This implementation should imitate the behaviour of the `rouge-score` package https://pypi.org/project/rouge-
    score/.

    Args:
        newline_sep:
            New line separate the inputs.
            This argument has not been in use any more. It is deprecated in v0.6 and will be removed in v0.7.
        use_stemmer:
            Use Porter stemmer to strip word suffixes to improve matching.
        rouge_keys:
            A list of rouge types to calculate.
            Keys that are allowed are ``rougeL``, ``rougeLsum``, and ``rouge1`` through ``rouge9``.
        decimal_places:
            The number of digits to round the computed the values to.
            This argument has not been in usd any more. It is deprecated in v0.6 and will be removed in v0.7.
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
            If the python packages ``nltk`` is not installed.
        ValueError:
            If any of the ``rouge_keys`` does not belong to the allowed set of keys.

    References:
        [1] ROUGE: A Package for Automatic Evaluation of Summaries by Chin-Yew Lin https://aclanthology.org/W04-1013/
    """

    def __init__(
        self,
        newline_sep: Optional[bool] = None,  # remove in v0.7
        use_stemmer: bool = False,
        rouge_keys: Union[str, Tuple[str, ...]] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),  # type: ignore
        decimal_places: Optional[bool] = None,  # remove in v0.7
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
        if newline_sep is not None:
            warnings.warn("Argument `newline_sep` is deprecated in v0.6 and will be removed in v0.7")
        if decimal_places is not None:
            warnings.warn("Argument `decimal_places` is deprecated in v0.6 and will be removed in v0.7")

        if use_stemmer or "rougeLsum" in rouge_keys:
            if not _NLTK_AVAILABLE:
                raise ValueError("Stemmer and/or `rougeLsum` requires that nltk is installed. Use `pip install nltk`.")
            import nltk

        if not isinstance(rouge_keys, tuple):
            rouge_keys = tuple([rouge_keys])
        for key in rouge_keys:
            if key not in ALLOWED_ROUGE_KEYS:
                raise ValueError(f"Got unknown rouge key {key}. Expected to be one of {ALLOWED_ROUGE_KEYS}")

        self.rouge_keys = rouge_keys
        self.rouge_keys_values = [ALLOWED_ROUGE_KEYS[key] for key in rouge_keys]
        self.stemmer = nltk.stem.porter.PorterStemmer() if use_stemmer else None

        # Adding stated dynamically to prevent IndexError during sync function as some lists can be empty.
        for rouge_key in self.rouge_keys:
            for score in ["fmeasure", "precision", "recall"]:
                self.add_state(f"{rouge_key}_{score}", [], dist_reduce_fx=None)

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

        output: Dict[Union[int, str], List[Dict[str, Tensor]]] = _rouge_score_update(
            preds, targets, self.rouge_keys_values, stemmer=self.stemmer
        )
        for rouge_key, metrics in output.items():
            for metric in metrics:
                for type, value in metric.items():
                    getattr(self, f"rouge{rouge_key}_{type}").append(value.to(self.device))

    def compute(self) -> Dict[str, Tensor]:
        """Calculate (Aggregate and provide confidence intervals) ROUGE score.

        Return:
            Python dictionary of rouge scores for each input rouge key.
        """
        update_output = {}
        for rouge_key in self.rouge_keys_values:
            for type in ["fmeasure", "precision", "recall"]:
                update_output[f"rouge{rouge_key}_{type}"] = getattr(self, f"rouge{rouge_key}_{type}")

        return _rouge_score_compute(update_output)

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

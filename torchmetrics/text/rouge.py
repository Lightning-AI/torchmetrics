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
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

from torch import Tensor
from typing_extensions import Literal

from torchmetrics import Metric
from torchmetrics.functional.text.rouge import (
    ALLOWED_ACCUMULATE_VALUES,
    ALLOWED_ROUGE_KEYS,
    _rouge_score_compute,
    _rouge_score_update,
)
from torchmetrics.utilities.imports import _NLTK_AVAILABLE

__doctest_requires__ = {("ROUGEScore",): ["nltk"]}


class ROUGEScore(Metric):
    """`Calculate Rouge Score`_, used for automatic summarization.

    This implementation should imitate the behaviour of the `rouge-score` package `Python ROUGE Implementation`

    Args:
        use_stemmer: Use Porter stemmer to strip word suffixes to improve matching.
        normalizer: A user's own normalizer function.
            If this is ``None``, replacing any non-alpha-numeric characters with spaces is default.
            This function must take a ``str`` and return a ``str``.
        tokenizer:
            A user's own tokenizer function. If this is ``None``, spliting by spaces is default
            This function must take a `str` and return ``Sequence[str]``
        accumulate:
            Useful in case of multi-reference rouge score.

            - ``avg`` takes the avg of all references with respect to predictions
            - ``best`` takes the best fmeasure score obtained between prediction and multiple corresponding references.

        rouge_keys: A list of rouge types to calculate.
            Keys that are allowed are ``rougeL``, ``rougeLsum``, and ``rouge1`` through ``rouge9``.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from torchmetrics.text.rouge import ROUGEScore
        >>> preds = "My name is John"
        >>> target = "Is your name John"
        >>> rouge = ROUGEScore()
        >>> from pprint import pprint
        >>> pprint(rouge(preds, target))
        {'rouge1_fmeasure': tensor(0.7500),
         'rouge1_precision': tensor(0.7500),
         'rouge1_recall': tensor(0.7500),
         'rouge2_fmeasure': tensor(0.),
         'rouge2_precision': tensor(0.),
         'rouge2_recall': tensor(0.),
         'rougeL_fmeasure': tensor(0.5000),
         'rougeL_precision': tensor(0.5000),
         'rougeL_recall': tensor(0.5000),
         'rougeLsum_fmeasure': tensor(0.5000),
         'rougeLsum_precision': tensor(0.5000),
         'rougeLsum_recall': tensor(0.5000)}


    Raises:
        ValueError:
            If the python packages ``nltk`` is not installed.
        ValueError:
            If any of the ``rouge_keys`` does not belong to the allowed set of keys.

    References:
        [1] ROUGE: A Package for Automatic Evaluation of Summaries by Chin-Yew Lin `Rouge Detail`_
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    def __init__(
        self,
        use_stemmer: bool = False,
        normalizer: Callable[[str], str] = None,
        tokenizer: Callable[[str], Sequence[str]] = None,
        accumulate: Literal["avg", "best"] = "best",
        rouge_keys: Union[str, Tuple[str, ...]] = ("rouge1", "rouge2", "rougeL", "rougeLsum"),  # type: ignore
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if use_stemmer or "rougeLsum" in rouge_keys:
            if not _NLTK_AVAILABLE:
                raise ModuleNotFoundError(
                    "Stemmer and/or `rougeLsum` requires that `nltk` is installed. Use `pip install nltk`."
                )
            import nltk

        if not isinstance(rouge_keys, tuple):
            rouge_keys = (rouge_keys,)
        for key in rouge_keys:
            if key not in ALLOWED_ROUGE_KEYS:
                raise ValueError(f"Got unknown rouge key {key}. Expected to be one of {ALLOWED_ROUGE_KEYS}")

        if accumulate not in ALLOWED_ACCUMULATE_VALUES:
            raise ValueError(
                f"Got unknown accumulate value {accumulate}. Expected to be one of {ALLOWED_ACCUMULATE_VALUES}"
            )

        self.rouge_keys = rouge_keys
        self.rouge_keys_values = [ALLOWED_ROUGE_KEYS[key] for key in rouge_keys]
        self.stemmer = nltk.stem.porter.PorterStemmer() if use_stemmer else None
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.accumulate = accumulate

        # Adding stated dynamically to prevent IndexError during sync function as some lists can be empty.
        for rouge_key in self.rouge_keys:
            for score in ["fmeasure", "precision", "recall"]:
                self.add_state(f"{rouge_key}_{score}", [], dist_reduce_fx=None)

    def update(  # type: ignore
        self, preds: Union[str, Sequence[str]], target: Union[str, Sequence[str], Sequence[Sequence[str]]]
    ) -> None:
        """Compute rouge scores.

        Args:
            preds: An iterable of predicted sentences or a single predicted sentence.
            target: An iterable of target sentences or an iterable of target sentences or a single target sentence.
        """
        if isinstance(target, list) and all(isinstance(tgt, str) for tgt in target):
            target = [target] if isinstance(preds, str) else [[tgt] for tgt in target]

        if isinstance(preds, str):
            preds = [preds]

        if isinstance(target, str):
            target = [[target]]

        output: Dict[Union[int, str], List[Dict[str, Tensor]]] = _rouge_score_update(
            preds,
            target,
            self.rouge_keys_values,
            stemmer=self.stemmer,
            normalizer=self.normalizer,
            tokenizer=self.tokenizer,
            accumulate=self.accumulate,
        )
        for rouge_key, metrics in output.items():
            for metric in metrics:
                for tp, value in metric.items():
                    getattr(self, f"rouge{rouge_key}_{tp}").append(value.to(self.device))

    def compute(self) -> Dict[str, Tensor]:
        """Calculate (Aggregate and provide confidence intervals) ROUGE score.

        Return:
            Python dictionary of rouge scores for each input rouge key.
        """
        update_output = {}
        for rouge_key in self.rouge_keys_values:
            for tp in ["fmeasure", "precision", "recall"]:
                update_output[f"rouge{rouge_key}_{tp}"] = getattr(self, f"rouge{rouge_key}_{tp}")

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

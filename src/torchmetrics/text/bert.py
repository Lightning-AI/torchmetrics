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
from typing import Any, Callable, Dict, List, Optional, Union
from warnings import warn

import torch
from torch import Tensor
from torch.nn import Module

from torchmetrics.functional.text.bert import _preprocess_text, bert_score
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TRANSFORMERS_AUTO_AVAILABLE

if _TRANSFORMERS_AUTO_AVAILABLE:
    from transformers.models.auto import AutoTokenizer
else:
    __doctest_skip__ = ["BERTScore"]


# Default model recommended in the original implementation.
_DEFAULT_MODEL = "roberta-large"


def _get_input_dict(input_ids: List[Tensor], attention_mask: List[Tensor]) -> Dict[str, Tensor]:
    """Create an input dictionary of ``input_ids`` and ``attention_mask`` for BERTScore calculation."""
    output_dict = {"input_ids": torch.cat(input_ids), "attention_mask": torch.cat(attention_mask)}
    return output_dict


class BERTScore(Metric):
    """`Bert_score Evaluating Text Generation`_ leverages the pre-trained contextual embeddings from BERT and
    matches words in candidate and reference sentences by cosine similarity. It has been shown to correlate with
    human judgment on sentence-level and system-level evaluation. Moreover, BERTScore computes precision, recall,
    and F1 measure, which can be useful for evaluating different language generation tasks.

    This implemenation follows the original implementation from `BERT_score`_.

    Args:
        preds: An iterable of predicted sentences.
        target: An iterable of target sentences.
        model_type: A name or a model path used to load `transformers` pretrained model.
        num_layers: A layer of representation to use.
        all_layers:
            An indication of whether the representation from all model's layers should be used.
            If `all_layers = True`, the argument `num_layers` is ignored.
        model:  A user's own model. Must be of `torch.nn.Module` instance.
        user_tokenizer:
            A user's own tokenizer used with the own model. This must be an instance with the `__call__` method.
            This method must take an iterable of sentences (`List[str]`) and must return a python dictionary
            containing `"input_ids"` and `"attention_mask"` represented by `torch.Tensor`. It is up to the user's model
            of whether `"input_ids"` is a `torch.Tensor` of input ids or embedding vectors.
            This tokenizer must prepend an equivalent of `[CLS]` token and append an equivalent of `[SEP]` token
            as `transformers` tokenizer does.
        user_forward_fn:
            A user's own forward function used in a combination with `user_model`. This function must take `user_model`
            and a python dictionary of containing `"input_ids"` and `"attention_mask"` represented by `torch.Tensor`
            as an input and return the model's output represented by the single `torch.Tensor`.
        verbose: An indication of whether a progress bar to be displayed during the embeddings' calculation.
        idf: An indication whether normalization using inverse document frequencies should be used.
        device: A device to be used for calculation.
        max_length: A maximum length of input sequences. Sequences longer than `max_length` are to be trimmed.
        batch_size: A batch size used for model processing.
        num_threads: A number of threads to use for a dataloader.
        return_hash: An indication of whether the correspodning `hash_code` should be returned.
        lang: A language of input sentences.
        rescale_with_baseline:
            An indication of whether bertscore should be rescaled with a pre-computed baseline.
            When a pretrained model from `transformers` model is used, the corresponding baseline is downloaded
            from the original `bert-score` package from `BERT_score`_ if available.
            In other cases, please specify a path to the baseline csv/tsv file, which must follow the formatting
            of the files from `BERT_score`_.
        baseline_path: A path to the user's own local csv/tsv file with the baseline scale.
        baseline_url: A url path to the user's own  csv/tsv file with the baseline scale.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Returns:
        Python dictionary containing the keys `precision`, `recall` and `f1` with corresponding values.

    Example:
        >>> from torchmetrics.text.bert import BERTScore
        >>> preds = ["hello there", "general kenobi"]
        >>> target = ["hello there", "master kenobi"]
        >>> bertscore = BERTScore()
        >>> score = bertscore(preds, target)
        >>> from pprint import pprint
        >>> rounded_score = {k: [round(v, 3) for v in vv] for k, vv in score.items()}
        >>> pprint(rounded_score)
        {'f1': [1.0, 0.996], 'precision': [1.0, 0.996], 'recall': [1.0, 0.996]}
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    preds_input_ids: List[Tensor]
    preds_attention_mask: List[Tensor]
    target_input_ids: List[Tensor]
    target_attention_mask: List[Tensor]

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        num_layers: Optional[int] = None,
        all_layers: bool = False,
        model: Optional[Module] = None,
        user_tokenizer: Optional[Any] = None,
        user_forward_fn: Callable[[Module, Dict[str, Tensor]], Tensor] = None,
        verbose: bool = False,
        idf: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 512,
        batch_size: int = 64,
        num_threads: int = 4,
        return_hash: bool = False,
        lang: str = "en",
        rescale_with_baseline: bool = False,
        baseline_path: Optional[str] = None,
        baseline_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path or _DEFAULT_MODEL
        self.num_layers = num_layers
        self.all_layers = all_layers
        self.model = model
        self.user_forward_fn = user_forward_fn
        self.verbose = verbose
        self.idf = idf
        self.embedding_device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.return_hash = return_hash
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline
        self.baseline_path = baseline_path
        self.baseline_url = baseline_url
        self.preds: Dict[str, List[Tensor]] = {"input_ids": [], "attention_mask": []}
        self.target: Dict[str, List[Tensor]] = {"input_ids": [], "attention_mask": []}

        if user_tokenizer:
            self.tokenizer = user_tokenizer
            self.user_tokenizer = True
        else:
            if not _TRANSFORMERS_AUTO_AVAILABLE:
                raise ModuleNotFoundError(
                    "`BERTScore` metric with default tokenizers requires `transformers` package be installed."
                    " Either install with `pip install transformers>=4.0` or `pip install torchmetrics[text]`."
                )
            if model_name_or_path is None:
                warn(
                    "The argument `model_name_or_path` was not specified while it is required when the default"
                    " `transformers` model is used."
                    f" It will use the default recommended model - {_DEFAULT_MODEL!r}."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.user_tokenizer = False

        self.add_state("preds_input_ids", [], dist_reduce_fx="cat")
        self.add_state("preds_attention_mask", [], dist_reduce_fx="cat")
        self.add_state("target_input_ids", [], dist_reduce_fx="cat")
        self.add_state("target_attention_mask", [], dist_reduce_fx="cat")

    def update(self, preds: List[str], target: List[str]) -> None:  # type: ignore
        """Store predictions/references for computing BERT scores. It is necessary to store sentences in a
        tokenized form to ensure the DDP mode working.

        Args:
            preds: An iterable of predicted sentences.
            target: An iterable of reference sentences.
        """
        preds_dict = _preprocess_text(
            preds,
            self.tokenizer,
            self.max_length,
            truncation=False,
            sort_according_length=False,
            own_tokenizer=self.user_tokenizer,
        )
        target_dict = _preprocess_text(
            target,
            self.tokenizer,
            self.max_length,
            truncation=False,
            sort_according_length=False,
            own_tokenizer=self.user_tokenizer,
        )

        self.preds_input_ids.append(preds_dict["input_ids"])
        self.preds_attention_mask.append(preds_dict["attention_mask"])
        self.target_input_ids.append(target_dict["input_ids"])
        self.target_attention_mask.append(target_dict["attention_mask"])

    def compute(self) -> Dict[str, Union[List[float], str]]:
        """Calculate BERT scores.

        Return:
            Python dictionary containing the keys `precision`, `recall` and `f1` with corresponding values.
        """
        return bert_score(
            preds=_get_input_dict(self.preds_input_ids, self.preds_attention_mask),
            target=_get_input_dict(self.target_input_ids, self.target_attention_mask),
            model_name_or_path=self.model_name_or_path,
            num_layers=self.num_layers,
            all_layers=self.all_layers,
            model=self.model,
            user_tokenizer=self.tokenizer if self.user_tokenizer else None,
            user_forward_fn=self.user_forward_fn,
            verbose=self.verbose,
            idf=self.idf,
            device=self.embedding_device,
            max_length=self.max_length,
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            return_hash=self.return_hash,
            lang=self.lang,
            rescale_with_baseline=self.rescale_with_baseline,
            baseline_path=self.baseline_path,
            baseline_url=self.baseline_url,
        )

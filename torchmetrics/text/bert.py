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
from deprecate import deprecated

from torchmetrics.functional import bert_score
from torchmetrics.functional.text.bert import _preprocess_text
from torchmetrics.metric import Metric
from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoTokenizer


# Default model recommended in the original implementation.
_DEFAULT_MODEL = "roberta-large"


def _concatenate(d: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Concatenate list of tensors within a given dictionary."""
    output_dict: Dict[str, torch.Tensor] = {}
    for k, v in d.items():
        output_dict[k] = torch.cat(v)
    return output_dict


class BERTScore(Metric):
    """`Bert_score Evaluating Text Generation`_ leverages the pre-trained contextual embeddings from BERT and
    matches words in candidate and reference sentences by cosine similarity. It has been shown to correlate with
    human judgment on sentence-level and system-level evaluation. Moreover, BERTScore computes precision, recall,
    and F1 measure, which can be useful for evaluating different language generation tasks.

    This implemenation follows the original implementation from `BERT_score`_.

    Args:
        preds:
            An iterable of predicted sentences.
        target:
            An iterable of target sentences.
        model_type:
            A name or a model path used to load `transformers` pretrained model.
        num_layers:
            A layer of representation to use.
        all_layers:
            An indication of whether the representation from all model's layers should be used.
            If `all_layers = True`, the argument `num_layers` is ignored.
        model:
            A user's own model. Must be of `torch.nn.Module` instance.
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
        verbose:
            An indication of whether a progress bar to be displayed during the embeddings calculation.
        idf:
            An indication whether normalization using inverse document frequencies should be used.
        device:
            A device to be used for calculation.
        max_length:
            A maximum length of input sequences. Sequences longer than `max_length` are to be trimmed.
        batch_size:
            A batch size used for model processing.
        num_threads:
            A number of threads to use for a dataloader.
        return_hash:
            An indication of whether the correspodning `hash_code` should be returned.
        lang:
            A language of input sentences.
        rescale_with_baseline:
            An indication of whether bertscore should be rescaled with a pre-computed baseline.
            When a pretrained model from `transformers` model is used, the corresponding baseline is downloaded
            from the original `bert-score` package from `BERT_score`_ if available.
            In other cases, please specify a path to the baseline csv/tsv file, which must follow the formatting
            of the files from `BERT_score`_.
        baseline_path:
            A path to the user's own local csv/tsv file with the baseline scale.
        baseline_url:
            A url path to the user's own  csv/tsv file with the baseline scale.
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called.
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    Returns:
        Python dictionary containing the keys `precision`, `recall` and `f1` with corresponding values.

    Example:
        >>> from torchmetrics.text.bert import BERTScore
        >>> preds = ["hello there", "general kenobi"]
        >>> target = ["hello there", "master kenobi"]
        >>> bertscore = BERTScore()
        >>> bertscore(preds, target)  # doctest: +SKIP
        {'precision': [0.99..., 0.99...],
         'recall': [0.99..., 0.99...],
         'f1': [0.99..., 0.99...]}
    """

    higher_is_better = True

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        num_layers: Optional[int] = None,
        all_layers: bool = False,
        model: Optional[torch.nn.Module] = None,
        user_tokenizer: Optional[Any] = None,
        user_forward_fn: Callable[[torch.nn.Module, Dict[str, torch.Tensor]], torch.Tensor] = None,
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
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.model_name_or_path = model_name_or_path
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
        self.preds: Dict[str, List[torch.Tensor]] = {"input_ids": [], "attention_mask": []}
        self.target: Dict[str, List[torch.Tensor]] = {"input_ids": [], "attention_mask": []}

        if user_tokenizer:
            self.tokenizer = user_tokenizer
            self.user_tokenizer = True
        else:
            if not _TRANSFORMERS_AVAILABLE:
                raise ModuleNotFoundError(
                    "`BERTScore` metric with default tokenizers requires `transformers` package be installed."
                    " Either install with `pip install transformers>=4.0` or `pip install torchmetrics[text]`."
                )
            if not model_name_or_path:
                model_name_or_path = _DEFAULT_MODEL
                warn(
                    "The argument `model_name_or_path` was not specified while it is required when default "
                    " `transformers` model are used."
                    f"It is, therefore, used the default recommended model - {_DEFAULT_MODEL}."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.user_tokenizer = False

    @deprecated(
        args_mapping={"predictions": "preds", "references": "target"},
        target=True,
        deprecated_in="0.7",
        remove_in="0.8",
    )
    def update(self, preds: List[str], target: List[str]) -> None:  # type: ignore
        """Store predictions/references for computing BERT scores. It is necessary to store sentences in a
        tokenized form to ensure the DDP mode working.

        Args:
            preds:
                An iterable of predicted sentences.
            target:
                An iterable of reference sentences.

        .. deprecated:: v0.7
            Args:
                predictions:
                    This argument is deprecated in favor of  `preds` and will be removed in v0.8.
                references:
                    This argument is deprecated in favor of  `target` and will be removed in v0.8.
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
        self.preds["input_ids"].append(preds_dict["input_ids"])
        self.preds["attention_mask"].append(preds_dict["attention_mask"])
        self.target["input_ids"].append(target_dict["input_ids"])
        self.target["attention_mask"].append(target_dict["attention_mask"])

    def compute(self) -> Dict[str, Union[List[float], str]]:
        """Calculate BERT scores.

        Return:
            Python dictionary containing the keys `precision`, `recall` and `f1` with corresponding values.
        """
        return bert_score(
            preds=_concatenate(self.preds),
            target=_concatenate(self.target),
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

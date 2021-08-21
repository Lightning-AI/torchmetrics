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

import torch
from transformers import AutoTokenizer

from torchmetrics.functional import bert_score
from torchmetrics.functional.text.bert import _preprocess_text
from torchmetrics.metric import Metric


def _flatten(x: List[List[str]]) -> List[str]:
    """converts list of list to single list of strings."""
    return [e for y in x for e in y]


def _concatenate(d: Dict[str, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    output_dict: Dict[str, torch.Tensor] = {}
    for k, v in d.items():
        output_dict[k] = torch.cat(v)
    return output_dict


class BERTScore(Metric):
    """`BERTScore <https://arxiv.org/abs/1904.09675>`_ leverages the pre-trained contextual embeddings from BERT
    and matches words in candidate and reference sentences by cosine similarity. It has been shown to correlate
    with human judgment on sentence-level and system-level evaluation. Moreover, BERTScore computes precision,
    recall, and F1 measure, which can be useful for evaluating different language generation tasks.

    This implemenation follows the original implementation from https://github.com/Tiiiger/bert_score.

    Args:
        predictions:
            An iterable of predicted sentences.
        references:
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
        verbose:
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
            An indication of whether bertscore should be rescaled with pre-computed baseline
        baseline_path:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    Returns:
        Python dictionary containing the keys `precision`, `recall` and `f1` with corresponding values.

    Example:
        >>> predictions = ["hello there", "general kenobi"]
        >>> references = ["hello there", "master kenobi"]
        >>> bertscore = BERTScore()
        >>> bertscore.update(predictions=predictions,references=references)
        >>> bertscore.compute()  # doctest: +SKIP
        {'precision': [0.99..., 0.99...],
         'recall': [0.99..., 0.99...],
         'f1': [0.99..., 0.99...],
         'hashcode': '...'}
    """

    def __init__(
        self,
        lang: str = "en",
        model_name_or_path: Optional[str] = None,
        num_layers: Optional[int] = None,
        all_layers: bool = False,
        model: Optional[torch.nn.Module] = None,
        verbose: bool = False,
        idf: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 512,
        batch_size: int = 64,
        num_threads: int = 4,
        return_hash: bool = True,
        rescale_with_baseline: bool = False,
        baseline_path: Optional[str] = None,
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
        self.lang = lang
        self.model_name_or_path = model_name_or_path
        self.num_layers = num_layers
        self.all_layers = all_layers
        self.model = model
        self.verbose = verbose
        self.idf = idf
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.return_hash = return_hash
        self.rescale_with_baseline = rescale_with_baseline
        self.baseline_path = baseline_path
        self.predictions: Dict[str, List[torch.Tensor]] = {"input_ids": [], "attention_mask": []}
        self.references: Dict[str, List[torch.Tensor]] = {"input_ids": [], "attention_mask": []}

        if model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            # TODO: Add a support for the custom tokenizer
            raise ValueError(
                "For now, the argument `model_name_or_path` must be provided to initialize `transformers` tokenizer, "
                "which is required for storing of the tokenized input sentences. This is required to ensure the "
                "module will work with the DDP mode."
            )

    def update(self, predictions: List[str], references: List[str]) -> None:  # type: ignore
        """Store predictions/references for computing BERT scores. It is necessary to store sentences in a
        tokenized form to ensure the DDP mode working.

        Args:
            predictions:
                An iterable of predicted sentences.
            references:
                An iterable of predicted sentences.
        """
        predictions_dict = _preprocess_text(
            predictions, self.tokenizer, self.max_length, truncation=False, sort_according_length=False
        )
        references_dict = _preprocess_text(
            references, self.tokenizer, self.max_length, truncation=False, sort_according_length=False
        )
        self.predictions["input_ids"].append(predictions_dict["input_ids"])
        self.predictions["attention_mask"].append(predictions_dict["attention_mask"])
        self.references["input_ids"].append(references_dict["input_ids"])
        self.references["attention_mask"].append(references_dict["attention_mask"])

    def compute(self) -> Dict[str, List[float]]:
        """Calculate BERT scores.

        Return:
            Python dictionary containing the keys `precision`, `recall` and `f1` with corresponding values.
        """
        return bert_score(
            predictions=_concatenate(self.predictions),
            references=_concatenate(self.references),
            lang=self.lang,
            model_name_or_path=self.model_name_or_path,
            num_layers=self.num_layers,
            all_layers=self.all_layers,
            model=self.model,
            verbose=self.verbose,
            idf=self.idf,
            device=self.device,
            max_length=self.max_length,
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            return_hash=self.return_hash,
            rescale_with_baseline=self.rescale_with_baseline,
            baseline_path=self.baseline_path,
        )

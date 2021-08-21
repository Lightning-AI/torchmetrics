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
import math
import warnings
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoModel, AutoTokenizer


def _preprocess_text(text: List[str], tokenizer: Any, max_length: int = 512) -> Dict[str, torch.Tensor]:
    """Default text pre-processing function using `transformers` `AutoTokenizer` instance.

    Args:
        text:
            An iterable of sentences.
        tokenizer:
            `AutoTokenizer` instance from `transformers` package.
        max_length:
            A maximum sequence length.

    Return:
        A dictionary of tokenized sentences including input_ids and attention_mask.
    """
    tokenized_data = tokenizer(text, padding=True, max_length=max_length, truncation=True, return_tensors="pt")
    input_ids, attention_mask = _sort_data_according_length(tokenized_data.input_ids, tokenized_data.attention_mask)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _process_attention_mask_for_special_tokens(attention_mask: torch.Tensor) -> torch.Tensor:
    """Process attention mask to be zero for special [CLS] and [SEP] tokens as they're not included in a
    calculation.

    Args:
        attention_mask:
            An attention mask to be returned, for example, by a `transformers` tokenizer.

    Return:
        A processd attention mask.
    """
    # Make attention_mask zero for [CLS] token
    attention_mask[:, 0] = 0
    # Make attention_mask zero for [SEP] token
    sep_token_position = (attention_mask - 0.1).cumsum(-1).argmax(-1)
    attention_mask[torch.arange(attention_mask.size(0)).long(), sep_token_position] = 0
    return attention_mask


def _sort_data_according_length(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    sorted_indices = attention_mask.sum(1).argsort()
    input_ids = input_ids[sorted_indices]
    attention_mask = attention_mask[sorted_indices]
    return input_ids, attention_mask


def _input_data_collator(
    batch: Dict[str, torch.Tensor], device: Optional[Union[str, torch.device]]
) -> Dict[str, torch.Tensor]:
    """Helper function that trims model inputs to the longest sequence within the batch and put the input on the
    proper device."""
    max_len = int(batch["attention_mask"].sum(1).max().item())
    input_ids = batch["input_ids"][:, :max_len].to(device)
    attention_mask = batch["attention_mask"][:, :max_len].to(device)
    batch.update({"input_ids": input_ids, "attention_mask": attention_mask})
    return batch


def _output_data_collator(
    model_output: torch.Tensor, attention_mask: torch.Tensor, target_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper function that pads the model output and attention mask to the target length."""
    zeros_shape = list(model_output.shape)
    zeros_shape[2] = target_len - zeros_shape[2]
    model_output = torch.cat(
        [model_output, torch.zeros(zeros_shape, dtype=model_output.dtype).to(model_output.device)], dim=2
    )
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.zeros(zeros_shape[0], zeros_shape[2], dtype=attention_mask.dtype).to(attention_mask.device)
        ], dim=1
    )
    return model_output, attention_mask


class TextDataset(Dataset):
    def __init__(
        self,
        text: List[str],
        tokenizer: Any,
        max_length: int = 512,
        preprocess_text_fn: Callable[[List[str], Any, int], Dict[str, torch.Tensor]] = _preprocess_text,
        idf: bool = False,
        tokens_idf: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Args:
            text:
                An iterable of sentences.
            tokenizer:
                `AutoTokenizer` instance from `transformers` package.
            max_length:
                A maximum sequence length.
            preprocess_text_fn:
            idf:
            tokens_idf:
                Inverse document frequences (these should be calculated on reference sentences).
        """
        self.text = preprocess_text_fn(text, tokenizer, max_length)
        self.max_length = self.text["input_ids"].shape[1]
        self.num_sentences = len(text)
        self.idf = idf
        if idf:
            self.tokens_idf = tokens_idf if tokens_idf is not None else self._get_tokens_idf()
        else:
            self.tokens_idf = {}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self.text["input_ids"][idx, :]
        attention_mask = self.text["attention_mask"][idx, :]
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.idf:
            input_ids_idf = torch.tensor([self.tokens_idf[input_idx] for input_idx in input_ids.tolist()])
            inputs_dict["input_ids_idf"] = input_ids_idf
        return inputs_dict

    def __len__(self) -> int:
        return self.num_sentences

    def _get_tokens_idf(self) -> Dict[int, float]:
        """Calculate token inverse document frequences.

        Return:
            A python dictionary containing inverse document frequences for token ids.
        """
        token_counter: Counter = Counter()
        for tokens in map(self._set_of_tokens, self.text["input_ids"]):
            token_counter.update(tokens)

        tokens_idf: Dict[int, float] = defaultdict(self._get_tokens_idf_default_value)
        tokens_idf.update(
            {
                idx: math.log((self.num_sentences + 1) / (occurrence + 1)) for idx, occurrence in token_counter.items()
            }
        )
        return tokens_idf

    def _get_tokens_idf_default_value(self) -> float:
        """Helper function that ensures `defaultdict` to be pickled."""
        return math.log((self.num_sentences + 1) / 1)

    @staticmethod
    def _set_of_tokens(input_ids: torch.Tensor) -> Set:
        return set(input_ids.tolist())


def _get_embeddings_and_idf_scale(
    dataloader: DataLoader,
    target_len: int,
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    num_layers: Optional[int] = None,
    all_layers: bool = False,
    idf: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate sentence embeddings and the inverse-document-frequence scaling factor.
    Args:
        dataloader:
            `torch.utils.data.DataLoader` instance.
        target_len:
            A length of the longest sequence in the data. Used for padding the model output.
        model:
            BERT model.
        device:
            A device to be used for calculation.
        num_layers:
            The layer of representation to use.
        all_layers:
            An indication whether representation from all model layers should be used for BERTScore.
        idf:
            An Indication whether normalization using inverse document frequencies should be used.

    Return:
    """
    EMBEDDINGS_LIST: List[torch.Tensor] = []
    IDF_SCALE_LIST: List[torch.Tensor] = []
    for batch in dataloader:
        with torch.no_grad():
            batch = _input_data_collator(batch, device)
            if not all_layers:
                # Output shape: batch_size x 1 x sequence_length x bert_dim
                out = (
                    model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        output_hidden_states=True,
                    )
                    .hidden_states[num_layers if num_layers is not None else -1]
                    .unsqueeze(1)
                )
            else:
                # Output shape: batch_size x num_layers x sequence_length x bert_dim
                out = model(batch["input_ids"], batch["attention_mask"], output_hidden_states=True).hidden_states
                out = torch.cat([o.unsqueeze(1) for o in out], dim=1)

        out /= out.norm(dim=-1).unsqueeze(-1)  # normalize embeddings
        out, attention_mask = _output_data_collator(out, batch["attention_mask"], target_len)
        processed_attention_mask = _process_attention_mask_for_special_tokens(attention_mask)
        # Multiply embeddings with attention_mask (b=batch_size, l=num_layers, s=seq_len, d=emb_dim)
        out = torch.einsum("blsd, bs -> blsd", out, processed_attention_mask)
        EMBEDDINGS_LIST.append(out.cpu())

        # Calculate weighted (w.r.t. sentence length) input_ids IDF matrix
        input_ids_idf = (
            batch["input_ids_idf"] * processed_attention_mask if idf else processed_attention_mask.type(out.dtype)
        )
        input_ids_idf /= input_ids_idf.sum(-1, keepdim=True)
        IDF_SCALE_LIST.append(input_ids_idf)

    EMBEDDINGS = torch.cat(EMBEDDINGS_LIST)
    IDF_SCALE = torch.cat(IDF_SCALE_LIST)

    return EMBEDDINGS, IDF_SCALE


def _get_scaled_precision_or_recall(cos_sim: torch.Tensor, metric: str, idf_scale: torch.Tensor) -> torch.Tensor:
    """Helper function that calculates precision or recall, transpose it and scale it with idf_scale factor."""
    dim = 3 if metric == "precision" else 2
    res = cos_sim.max(dim=dim).values
    res = torch.einsum("bls, bs -> bls", res, idf_scale).sum(-1)
    # We transpose the results and squeeze if possible to match the format of the original BERTScore implementation
    res = res.transpose(0, 1).squeeze()
    return res


def _get_precision_recall_f1(
    pred_embeddings: torch.Tensor,
    ref_embeddings: torch.Tensor,
    pred_idf_scale: torch.Tensor,
    ref_idf_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate precision, recall and F1 score over candidate and reference sentences.

    Args:
        pred_embeddings:
            Embeddings of candidate sentenecs.
        ref_embeddings:
            Embeddings of reference sentences.
        pred_idf_scale:
            An IDF scale factor for candidate sentences.
        ref_idf_scale:
            An IDF scale factor for reference sentences.

    Return:
        Tensors containing precision, recall and F1 score, respectively.
    """
    # Dimensions: b = batch_size, l = num_layers, p = predictions_seq_len, r = references_seq_len, d = bert_dim
    cos_sim = torch.einsum("blpd, blrd -> blpr", pred_embeddings, ref_embeddings)
    # Final metrics shape = (batch_size * num_layers | batch_size)
    precision = _get_scaled_precision_or_recall(cos_sim, "precision", pred_idf_scale)
    recall = _get_scaled_precision_or_recall(cos_sim, "recall", ref_idf_scale)

    f1_score = 2 * precision * recall / (precision + recall)
    f1_score = f1_score.masked_fill(torch.isnan(f1_score), 0.0)

    return precision, recall, f1_score


def bert_score(
    predictions: List[str],
    references: List[str],
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
    lang: str = "en",
    rescale_with_baseline: bool = False,
    baseline_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """`BERTScore <https://arxiv.org/abs/1904.09675>`_ leverages the pre-trained contextual embeddings from BERT
    and matches words in candidate and reference sentences by cosine similarity. It has been shown to correlate
    with human judgment on sentence-level and system-level evaluation. Moreover, BERTScore computes precision,
    recall, and F1 measure, which can be useful for evaluating different language generation tasks. assert
    len(predictions) == len(references), "Number of predicted and reference sententes must be the same!".

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
        lang:
            A language of input sentences.
        rescale_with_baseline:
            An indication of whether bertscore should be rescaled with pre-computed baseline
        baseline_path:

    Returns:
        Python dictionary containing the keys `precision`, `recall` and `f1` with corresponding values.

    Example:
        >>> predictions = ["hello there", "general kenobi"]
        >>> references = ["hello there", "master kenobi"]
        >>> bert_score(predictions=predictions, references=references, lang="en")  # doctest: +SKIP
        {'precision': [0.99..., 0.99...],
         'recall': [0.99..., 0.99...],
         'f1': [0.99..., 0.99...]}
    """
    assert len(predictions) == len(references), "Number of predicted and reference sententes must be the same!"

    if model is None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ValueError(
                "`bert_score` metric with default models requires `transformers` package be installed. "
                "Either install with `pip install transformers>=4.0` or `pip install torchmetrics[text]`"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    model.to(device)

    try:
        if num_layers:
            assert num_layers <= model.config.num_hidden_layers, (
                f"num_layers={num_layers} is forbidden for {model_name_or_path}. "
                f"Please use num_layers <= {model.config.num_hidden_layers}"
            )
    except AttributeError:
        warnings.warn("It was not possible to retrieve the parametery `num_layers` from the model specification.")

    ref_dataset = TextDataset(references, tokenizer, max_length, idf=idf)
    pred_dataset = TextDataset(predictions, tokenizer, max_length, idf=idf, tokens_idf=ref_dataset.tokens_idf)
    ref_loader = DataLoader(ref_dataset, batch_size=batch_size, num_workers=num_threads)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, num_workers=num_threads)

    ref_embeddings, ref_idf_scale = _get_embeddings_and_idf_scale(
        ref_loader, ref_dataset.max_length, model, device, num_layers, all_layers, idf
    )
    pred_embeddings, pred_idf_scale = _get_embeddings_and_idf_scale(
        pred_loader, pred_dataset.max_length, model, device, num_layers, all_layers, idf
    )

    precision, recall, f1_score = _get_precision_recall_f1(
        pred_embeddings, ref_embeddings, pred_idf_scale, ref_idf_scale
    )

    output_dict = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1_score.tolist(),
    }
    return output_dict

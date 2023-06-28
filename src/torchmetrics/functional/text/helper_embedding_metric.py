# Copyright The Lightning team.
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
import os
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
else:
    PreTrainedModel = PreTrainedTokenizerBase = None

if _TQDM_AVAILABLE:
    import tqdm


def _process_attention_mask_for_special_tokens(attention_mask: Tensor) -> Tensor:
    """Process attention mask to be zero for special [CLS] and [SEP] tokens as they're not included in BERT score.

    Args:
        attention_mask: An attention mask to be returned, for example, by a `transformers` tokenizer.

    Return:
        A processed attention mask.
    """
    # Make attention_mask zero for [CLS] token
    attention_mask[:, 0] = 0
    # Make attention_mask zero for [SEP] token
    sep_token_position = _cumsum((attention_mask - 0.1), dim=-1).argmax(-1)
    attention_mask[torch.arange(attention_mask.size(0)).long(), sep_token_position] = 0
    return attention_mask


def _input_data_collator(
    batch: Dict[str, Tensor], device: Optional[Union[str, torch.device]] = None
) -> Dict[str, Tensor]:
    """Trim model inputs.

    This function trims the model inputs to the longest sequence within the batch and put the input on the proper
    device.
    """
    max_len = int(batch["attention_mask"].sum(1).max().item())
    input_ids = batch["input_ids"][:, :max_len].to(device)
    attention_mask = batch["attention_mask"][:, :max_len].to(device)
    batch.update({"input_ids": input_ids, "attention_mask": attention_mask})
    return batch


def _output_data_collator(model_output: Tensor, attention_mask: Tensor, target_len: int) -> Tuple[Tensor, Tensor]:
    """Pad the model output and attention mask to the target length."""
    zeros_shape = list(model_output.shape)
    zeros_shape[2] = target_len - zeros_shape[2]
    model_output = torch.cat(
        [model_output, torch.zeros(zeros_shape, dtype=model_output.dtype).to(model_output.device)], dim=2
    )
    zeros = torch.zeros(zeros_shape[0], zeros_shape[2], dtype=attention_mask.dtype).to(attention_mask.device)
    attention_mask = torch.cat([attention_mask, zeros], dim=1)
    return model_output, attention_mask


def _sort_data_according_length(input_ids: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Sort tokenized sentence from the shortest to the longest one."""
    sorted_indices = attention_mask.sum(1).argsort()
    input_ids = input_ids[sorted_indices]
    attention_mask = attention_mask[sorted_indices]
    return input_ids, attention_mask, sorted_indices


def _preprocess_text(
    text: List[str],
    tokenizer: Any,
    max_length: int = 512,
    truncation: bool = True,
    sort_according_length: bool = True,
    own_tokenizer: bool = False,
) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
    """Text pre-processing function using `transformers` `AutoTokenizer` instance.

    Args:
        text:
            An iterable of sentences.
        tokenizer:
            Either `AutoTokenizer` instance from `transformers` package, or a user's own tokenizer.
        max_length:
            A maximum sequence length.
        truncation:
            An indication of whether tokenized sequences should be padded only to the length of the longest sequence.
        sort_according_length:
            An indication of whether tokenized sequences should be sorted from shortest to longest. This is appropriate
            to do for leveraging dynamic padding during embedding calculation and thereby to hasten inference.
        own_tokenizer:
            An indication of whether a non-default user's own tokenizer is used.

    Return:
        A dictionary of tokenized sentences including input_ids and attention_mask.

    Raises:
        BaseException:
            If a tokenization with a user's own tokenizer is not successful.
    """
    if not own_tokenizer:
        tokenized_data = tokenizer(
            text, padding="max_length", max_length=max_length, truncation=truncation, return_tensors="pt"
        )
    else:
        try:
            tokenized_data = tokenizer(text, max_length)
        except BaseException as ex:
            raise RuntimeError(f"Tokenization was not successful: {ex}") from ex

    if sort_according_length:
        input_ids, attention_mask, sorting_indices = _sort_data_according_length(
            tokenized_data["input_ids"], tokenized_data["attention_mask"]
        )
        input_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
    else:
        input_dict = {"input_ids": tokenized_data["input_ids"], "attention_mask": tokenized_data["attention_mask"]}
        sorting_indices = None

    return input_dict, sorting_indices


def _get_progress_bar(dataloader: DataLoader, verbose: bool = False) -> Union[DataLoader, "tqdm.auto.tqdm"]:
    """Wrap dataloader in progressbar if asked for.

    Function will return either the dataloader itself when `verbose = False`, or it wraps the dataloader with
    `tqdm.auto.tqdm`, when `verbose = True` to display a progress bar during the embbeddings calculation.
    """
    return tqdm.auto.tqdm(dataloader) if verbose else dataloader


def _check_shape_of_model_output(output: Tensor, input_ids: Tensor) -> None:
    """Check if the shape of the user's own model output."""
    bs, seq_len = input_ids.shape[:2]
    invalid_out_shape = len(output.shape) != 3 or output.shape[0] != bs or output.shape[1] != seq_len
    if invalid_out_shape:
        raise ValueError(
            "The model output must be `Tensor` of a shape `[batch_size, seq_len, model_dim]` "
            f"i.e. [{bs}, {seq_len}. , `model_dim`], but got {output.shape}."
        )


def _load_tokenizer_and_model(
    model_name_or_path: Union[str, os.PathLike], device: Optional[Union[str, torch.device]] = None
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load HuggingFace `transformers`' tokenizer and model. This function also handle a device placement.

    Args:
        model_name_or_path:
            A name or a model path used to load `transformers` pretrained model.
        device:
            A device to be used for calculation.

    Return:
        Initialized `transformers`' tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    model.eval()
    model.to(device)
    return tokenizer, model


class TextDataset(Dataset):
    """PyTorch dataset class for storing tokenized sentences and other properties used for BERT score calculation."""

    def __init__(
        self,
        text: List[str],
        tokenizer: Any,
        max_length: int = 512,
        preprocess_text_fn: Callable[
            [List[str], Any, int], Union[Dict[str, Tensor], Tuple[Dict[str, Tensor], Optional[Tensor]]]
        ] = _preprocess_text,
        idf: bool = False,
        tokens_idf: Optional[Dict[int, float]] = None,
    ) -> None:
        """Initialize text dataset class.

        Args:
            text: An iterable of sentences.
            tokenizer: `AutoTokenizer` instance from `transformers` package.
            max_length: A maximum sequence length.
            preprocess_text_fn: A function used for processing the input sentences.
            idf: An indication of whether calculate token inverse document frequencies to weight the model embeddings.
            tokens_idf: Inverse document frequencies (these should be calculated on reference sentences).
        """
        _text = preprocess_text_fn(text, tokenizer, max_length)
        if isinstance(_text, tuple):
            self.text, self.sorting_indices = _text
        else:
            self.text = _text
        self.max_length = self.text["input_ids"].shape[1]
        self.num_sentences = len(text)
        self.idf = idf
        self.tokens_idf = {}
        if idf:
            self.tokens_idf = tokens_idf if tokens_idf is not None else self._get_tokens_idf()

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get the input ids and attention mask belonging to a specific datapoint."""
        input_ids = self.text["input_ids"][idx, :]
        attention_mask = self.text["attention_mask"][idx, :]
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.idf:
            input_ids_idf = torch.tensor([self.tokens_idf[input_idx] for input_idx in input_ids.tolist()])
            inputs_dict["input_ids_idf"] = input_ids_idf
        return inputs_dict

    def __len__(self) -> int:
        """Return the number of sentences in the dataset."""
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
            {idx: math.log((self.num_sentences + 1) / (occurrence + 1)) for idx, occurrence in token_counter.items()}
        )
        return tokens_idf

    def _get_tokens_idf_default_value(self) -> float:
        """Ensure `defaultdict` can be pickled."""
        return math.log((self.num_sentences + 1) / 1)

    @staticmethod
    def _set_of_tokens(input_ids: Tensor) -> Set:
        """Return set of tokens from the `input_ids` :class:`~torch.Tensor`."""
        return set(input_ids.tolist())


class TokenizedDataset(TextDataset):
    """The child class of `TextDataset` class used with already tokenized data."""

    def __init__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        idf: bool = False,
        tokens_idf: Optional[Dict[int, float]] = None,
    ) -> None:
        """Initialize the dataset class.

        Args:
            input_ids: Input indexes
            attention_mask: Attention mask
            idf:
                An indication of whether calculate token inverse document frequencies to weight the model embeddings.
            tokens_idf: Inverse document frequencies (these should be calculated on reference sentences).
        """
        text = dict(
            zip(
                ["input_ids", "attention_mask", "sorting_indices"],
                _sort_data_according_length(input_ids, attention_mask),
            )
        )
        self.sorting_indices = text.pop("sorting_indices")
        self.text = _input_data_collator(text)
        self.num_sentences = len(self.text["input_ids"])
        self.max_length = self.text["input_ids"].shape[1]
        self.idf = idf
        self.tokens_idf = {}
        if idf:
            self.tokens_idf = tokens_idf if tokens_idf is not None else self._get_tokens_idf()

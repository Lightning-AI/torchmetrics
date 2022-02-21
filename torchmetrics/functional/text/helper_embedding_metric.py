import math
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


def _sort_data_according_length(input_ids: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Sort tokenized sentence from the shortest to the longest one."""
    sorted_indices = attention_mask.sum(1).argsort()
    input_ids = input_ids[sorted_indices]
    attention_mask = attention_mask[sorted_indices]
    return input_ids, attention_mask

def _preprocess_text(
    text: List[str],
    tokenizer: Any,
    max_length: int = 512,
    truncation: bool = True,
    sort_according_length: bool = True,
    own_tokenizer: bool = False,
) -> Dict[str, Tensor]:
    """Default text pre-processing function using `transformers` `AutoTokenizer` instance.

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
        except BaseException as e:
            raise BaseException(f"Tokenization was not successful: {e}")

    input_ids, attention_mask = (
        _sort_data_according_length(tokenized_data["input_ids"], tokenized_data["attention_mask"])
        if sort_according_length
        else (tokenized_data["input_ids"], tokenized_data["attention_mask"])
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask}


class TextDataset(Dataset):
    """PyTorch dataset class for storing tokenized sentences and other properties used for BERT score
    calculation."""

    def __init__(
        self,
        text: List[str],
        tokenizer: Any,
        max_length: int = 512,
        preprocess_text_fn: Callable[[List[str], Any, int], Dict[str, Tensor]] = _preprocess_text,
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
                A function used for processing the input sentences.
            idf:
                An indication of whether calculate token inverse document frequencies to weight the model embeddings.
            tokens_idf:
                Inverse document frequencies (these should be calculated on reference sentences).
        """
        self.text = preprocess_text_fn(text, tokenizer, max_length)
        self.max_length = self.text["input_ids"].shape[1]
        self.num_sentences = len(text)
        self.idf = idf
        self.tokens_idf = {}
        if idf:
            self.tokens_idf = tokens_idf if tokens_idf is not None else self._get_tokens_idf()

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
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
            {idx: math.log((self.num_sentences + 1) / (occurrence + 1)) for idx, occurrence in token_counter.items()}
        )
        return tokens_idf

    def _get_tokens_idf_default_value(self) -> float:
        """Helper function that ensures `defaultdict` to be pickled."""
        return math.log((self.num_sentences + 1) / 1)

    @staticmethod
    def _set_of_tokens(input_ids: Tensor) -> Set:
        """Return set of tokens from the `input_ids` `torch.Tensor`."""
        return set(input_ids.tolist())

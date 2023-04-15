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
import os
from enum import unique
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader
from typing_extensions import Literal

from torchmetrics.functional.text.helper_embedding_metric import (
    TokenizedDataset,
    _get_progress_bar,
    _input_data_collator,
    _load_tokenizer_and_model,
)
from torchmetrics.utilities.enums import EnumStr
from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
else:
    PreTrainedModel = PreTrainedTokenizerBase = None
    __doctest_skip__ = ["infolm"]


_ALLOWED_INFORMATION_MEASURE_LITERAL = Literal[
    "kl_divergence",
    "alpha_divergence",
    "beta_divergence",
    "ab_divergence",
    "renyi_divergence",
    "l1_distance",
    "l2_distance",
    "l_infinity_distance",
    "fisher_rao_distance",
]


@unique
class _IMEnum(EnumStr):
    """A helper Enum class for storing the information measure."""

    @staticmethod
    def _name() -> str:
        return "Information measure"

    KL_DIVERGENCE = "kl_divergence"
    ALPHA_DIVERGENCE = "alpha_divergence"
    BETA_DIVERGENCE = "beta_divergence"
    AB_DIVERGENCE = "ab_divergence"
    RENYI_DIVERGENCE = "renyi_divergence"
    L1_DISTANCE = "l1_distance"
    L2_DISTANCE = "l2_distance"
    L_INFINITY_DISTANCE = "l_infinity_distance"
    FISHER_RAO_DISTANCE = "fisher_rao_distance"


class _InformationMeasure:
    """A wrapper class used for the calculation of different information measures.

    This metric can be used to measure the information between the discrete reference distributions of predicted and
    reference sentences. The class also handles input validation for `alpha` and `beta` parameters.

    Args:
        information_measure:
            A name of information measure to be used. Please use one of: ['kl_divergence', 'alpha_divergence',
            'beta_divergence', 'ab_divergence', 'renyi_divergence', 'l1_distance', 'l2_distance', 'l_infinity_distance',
            'fisher_rao_distance']
        alpha:
            Alpha parameter of the divergence used for alpha, AB and Rényi divergence measures.
        beta:
            Beta parameter of the divergence used for beta and AB divergence measures.

    Raises:
        ValueError:
            If information measure is one from alpha, AB or Rényi divergence and parameter `alpha` is `None`.
        ValueError:
            If information measure is one from beta or divergence and parameter `beta` is `None`.
        ValueError:
            If information measure is alpha divergence and parameter `alpha` equals 0 or 1.
        ValueError:
            If information measure is beta divergence and parameter `beta` equals 0 or -1.
        ValueError:
            If information measure is AB divergence and parameter `alpha`, `beta` or `alpha + beta` equal 0.
        ValueError:
            If information measure is Rényi divergence and parameter `alpha` equals 1.
    """

    def __init__(
        self,
        information_measure: _ALLOWED_INFORMATION_MEASURE_LITERAL,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        self.information_measure = _IMEnum.from_str(information_measure)
        _bad_measures = (_IMEnum.ALPHA_DIVERGENCE, _IMEnum.AB_DIVERGENCE, _IMEnum.RENYI_DIVERGENCE)
        if self.information_measure in _bad_measures and not isinstance(alpha, float):
            raise ValueError(f"Parameter `alpha` is expected to be defined for {information_measure}.")
        if self.information_measure in [_IMEnum.BETA_DIVERGENCE, _IMEnum.AB_DIVERGENCE] and not isinstance(beta, float):
            raise ValueError(f"Parameter `beta` is expected to be defined for {information_measure}.")
        if self.information_measure == _IMEnum.ALPHA_DIVERGENCE and (not isinstance(alpha, float) or alpha in [0, 1]):
            raise ValueError(
                f"Parameter `alpha` is expected to be float differened from 0 and 1 for {information_measure}."
            )
        if self.information_measure == _IMEnum.BETA_DIVERGENCE and (not isinstance(beta, float) or beta in [0, -1]):
            raise ValueError(
                f"Parameter `beta` is expected to be float differened from 0 and -1 for {information_measure}."
            )
        if self.information_measure == _IMEnum.AB_DIVERGENCE and (
            alpha is None
            or beta is None
            or (any(not isinstance(p, float) for p in [alpha, beta]) or 0 in [alpha, beta, alpha + beta])
        ):
            raise ValueError(
                "Parameters `alpha`, `beta` and their sum are expected to be differened from 0 for "
                f"{information_measure}."
            )
        if self.information_measure == _IMEnum.RENYI_DIVERGENCE and (not isinstance(alpha, float) or alpha == 1):
            raise ValueError(f"Parameter `alpha` is expected to be float differened from 1 for {information_measure}.")

        # We ensure self.alpha and self.beta to be different from None to ensure mypy compliance
        self.alpha = alpha or 0
        self.beta = beta or 0

    def __call__(self, preds_distribution: Tensor, target_distribtuion: Tensor) -> Tensor:
        information_measure_function = getattr(self, f"_calculate_{self.information_measure.value}")
        return torch.nan_to_num(information_measure_function(preds_distribution, target_distribtuion))

    @staticmethod
    def _calculate_kl_divergence(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate Kullback-Leibler divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Kullback-Leibler divergence between discrete distributions of predicted and reference sentences.
        """
        return torch.sum(target_distribution * torch.log(preds_distribution / target_distribution), dim=-1)

    def _calculate_alpha_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate alpha divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Alpha divergence between discrete distributions of predicted and reference sentences.
        """
        _alpha_denom = self.alpha * (self.alpha - 1)
        return (
            1 - torch.sum(target_distribution**self.alpha * preds_distribution ** (1 - self.alpha), dim=-1)
        ) / _alpha_denom

    def _calculate_ab_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate AB divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            AB divergence between discrete distributions of predicted and reference sentences.
        """
        a = torch.log(torch.sum(target_distribution ** (self.beta + self.alpha), dim=-1))
        a /= self.beta * (self.beta + self.alpha)
        b = torch.log(torch.sum(preds_distribution ** (self.beta + self.alpha), dim=-1))
        b /= self.alpha * (self.beta + self.alpha)
        c = torch.log(torch.sum(target_distribution**self.alpha * preds_distribution**self.beta, dim=-1))
        c /= self.alpha * self.beta

        return a + b - c

    def _calculate_beta_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate beta divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Beta divergence between discrete distributions of predicted and reference sentences.
        """
        self.alpha = 1.0
        return self._calculate_ab_divergence(preds_distribution, target_distribution)

    def _calculate_renyi_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate Rényi divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Rényi divergence between discrete distributions of predicted and reference sentences.
        """
        return (
            torch.log(torch.sum(target_distribution**self.alpha * preds_distribution ** (1 - self.alpha), dim=-1))
        ) / (self.alpha - 1)

    @staticmethod
    def _calculate_l1_distance(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate L1 distance between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            L1 distance between discrete distributions of predicted and reference sentences.
        """
        return torch.norm(target_distribution - preds_distribution, p=1, dim=-1)

    @staticmethod
    def _calculate_l2_distance(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate L2 distance between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            L2 distance between discrete distributions of predicted and reference sentences.
        """
        return torch.norm(target_distribution - preds_distribution, p=2, dim=-1)

    @staticmethod
    def _calculate_l_infinity_distance(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate L-infinity distance between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            L-infinity distance between discrete distributions of predicted and reference sentences.
        """
        return torch.norm(target_distribution - preds_distribution, p=float("inf"), dim=-1)

    @staticmethod
    def _calculate_fisher_rao_distance(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate Fisher-Rao distance between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Fisher-Rao distance between discrete distributions of predicted and reference sentences.
        """
        return 2 * torch.acos(torch.clamp(torch.sqrt(preds_distribution * target_distribution).sum(-1), 0, 1))


def _get_dataloader(
    input_ids: Tensor, attention_mask: Tensor, idf: bool, batch_size: int, num_workers: int
) -> DataLoader:
    """Prepare dataloader.

    Args:
        input_ids:
            Indices of input sequence tokens in the vocabulary.
        attention_mask:
            Mask to avoid performing attention on padding token indices.
        idf:
            A bool indicating whether normalization using inverse document frequencies should be used.
        batch_size:
            A batch size used for model processing.
        num_workers:
            A number of workers to use for a dataloader.

    Return:
        An instance of ``torch.utils.data.DataLoader`` used for iterating over examples.
    """
    dataset = TokenizedDataset(input_ids, attention_mask, idf)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def _get_special_tokens_map(tokenizer: PreTrainedTokenizerBase) -> Dict[str, int]:
    """Build a dictionary of model/tokenizer special tokens.

    Args:
        tokenizer:
            Initialized tokenizer from HuggingFace's `transformers package.

    Return:
        A dictionary containing: mask_token_id, pad_token_id, sep_token_id and cls_token_id.
    """
    return {
        "mask_token_id": tokenizer.mask_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "sep_token_id": tokenizer.sep_token_id,
        "cls_token_id": tokenizer.cls_token_id,
    }


def _get_token_mask(input_ids: Tensor, pad_token_id: int, sep_token_id: int, cls_token_id: int) -> Tensor:
    """Generate a token mask for differentiating all special tokens in the input batch.

    There are 0s for special tokens and 1s otherwise.

    Args:
        input_ids:
            Indices of input sequence tokens in the vocabulary.
        pad_token_id:
            An id of ``<PAD>`` tokens that are used to make arrays of tokens the same size for batching purpose
        cls_token_id:
            An id of ``<CLS>`` token that represents the class of the input. (It might be ``<BOS>`` token for some
            models.)
        sep_token_id:
            An id of ``<SEP>`` token that separates two different sentences in the same input. (It might be ``<EOS>``
            token for some models.)

    Return:
        Tensor mask of 0s and 1s that masks all special tokens in the ``input_ids`` tensor.
    """
    token_mask = input_ids.eq(pad_token_id) | input_ids.eq(sep_token_id) | input_ids.eq(cls_token_id)
    return ~token_mask


def _get_batch_distribution(
    model: PreTrainedModel, batch: Dict[str, Tensor], temperature: float, idf: bool, special_tokens_map: Dict[str, int]
) -> Tensor:
    """Calculate a discrete probability distribution for a batch of examples. See `InfoLM`_ for details.

    Args:
        model:
            Initialized model from HuggingFace's `transformers package.
        batch:
            An input batch dictionary containing ``input_ids`` and ``attention_mask``.
        temperature:
            A temperature for calibrating language modelling. For more information, please reference `InfoLM`_ paper.
        max_length:
            A maximum length of input sequences. Sequences longer than `max_length` are to be trimmed.
        idf:
            An indication of whether normalization using inverse document frequencies should be used.
        special_tokens_map:
            A dictionary mapping tokenizer special tokens into the corresponding integer values.

    Return:
        A discrete probability distribution.
    """
    seq_len = batch["input_ids"].shape[1]
    prob_distribution_batch_list: List[Tensor] = []
    token_mask = _get_token_mask(
        batch["input_ids"],
        special_tokens_map["pad_token_id"],
        special_tokens_map["sep_token_id"],
        special_tokens_map["cls_token_id"],
    )
    for mask_idx in range(seq_len):
        input_ids = batch["input_ids"].clone()
        input_ids[:, mask_idx] = special_tokens_map["mask_token_id"]
        logits_distribution = model(input_ids, batch["attention_mask"]).logits
        # [batch_size, seq_len, vocab_size] -> [batch_size, vocab_size]
        logits_distribution = logits_distribution[:, mask_idx, :]
        prob_distribution = F.softmax(logits_distribution / temperature, dim=-1)
        if idf:
            prob_distribution *= batch["input_ids_idf"][:, mask_idx].unsqueeze(1).to(prob_distribution.device)
        prob_distribution_batch_list.append(prob_distribution.unsqueeze(1).cpu())  # [batch_size, 1, vocab_size]
        # Clean from memory
        del input_ids, logits_distribution, prob_distribution

    prob_distribution_batch = torch.cat(prob_distribution_batch_list, dim=1)  # [batch_size, seq_len, vocab_size]
    prob_distribution_batch = torch.einsum("bsv, bs -> bsv", prob_distribution_batch.to(token_mask.device), token_mask)
    if idf:
        masked_input_ids_idf = token_mask * batch["input_ids_idf"].to(token_mask.device)
        return prob_distribution_batch.sum(dim=1) / masked_input_ids_idf.sum(dim=1).unsqueeze(1)

    return prob_distribution_batch.sum(dim=1) / token_mask.sum(dim=1).unsqueeze(1)


@torch.no_grad()
def _get_data_distribution(
    model: PreTrainedModel,
    dataloader: DataLoader,
    temperature: float,
    idf: bool,
    special_tokens_map: Dict[str, int],
    verbose: bool,
) -> Tensor:
    """Calculate a discrete probability distribution according to the methodology described in `InfoLM`_.

    Args:
        model:
            Initialized model from HuggingFace's `transformers package.
        dataloader:
            An instance of `torch.utils.data.DataLoader` used for iterating over examples.
        temperature:
            A temperature for calibrating language modelling. For more information, please reference `InfoLM`_ paper.
        max_length:
            A maximum length of input sequences. Sequences longer than `max_length` are to be trimmed.
        idf:
            An indication of whether normalization using inverse document frequencies should be used.
        special_tokens_map:
            A dictionary mapping tokenizer special tokens into the corresponding integer values.
        verbose:
            An indication of whether a progress bar to be displayed during the embeddings calculation.

    Return:
        A discrete probability distribution.
    """
    device = model.device
    prob_distribution: List[Tensor] = []

    for batch in _get_progress_bar(dataloader, verbose):
        batch = _input_data_collator(batch, device)
        prob_distribution.append(_get_batch_distribution(model, batch, temperature, idf, special_tokens_map))

    return torch.cat(prob_distribution, dim=0)


def _infolm_update(
    preds: Union[str, Sequence[str]],
    target: Union[str, Sequence[str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Update the metric state by a tokenization of ``preds`` and ``target`` sentencens.

    Args:
        preds:
            An iterable of hypothesis corpus.
        target:
            An iterable of reference corpus.
        tokenizer:
            Initialized tokenizer from HuggingFace's `transformers package.
        max_length:
            A maximum length of input sequences. Sequences longer than `max_length` are to be trimmed.

    Return:
        Tokenizerd ``preds`` and ``target`` sentences represented with ``input_ids`` and ``attention_mask`` tensors.
    """
    # HuggingFace tokenizer expects an input to be of a type str or List[str]
    if not isinstance(preds, (str, list)):
        preds = list(preds)
    if not isinstance(target, (str, list)):
        target = list(target)

    preds_input = tokenizer(preds, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    target_input = tokenizer(target, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")

    return preds_input.input_ids, preds_input.attention_mask, target_input.input_ids, target_input.attention_mask


def _infolm_compute(
    model: PreTrainedModel,
    preds_dataloader: DataLoader,
    target_dataloader: DataLoader,
    temperature: float,
    idf: bool,
    information_measure_cls: _InformationMeasure,
    special_tokens_map: Dict[str, int],
    verbose: bool = True,
) -> Tensor:
    """Calculate selected information measure using the pre-trained language model.

    Args:
        model:
            Initialized model from HuggingFace's `transformers package.
        preds_dataloader:
            Loader iterating over tokenizer predicted sentences.
        target_dataloader:
            Loader iterating over tokenizer reference sentences.
        temperature:
            A temperature for calibrating language modelling. For more information, please reference `InfoLM`_ paper.
        idf:
            An indication of whether normalization using inverse document frequencies should be used.
        information_measure_cls:
            Information measure class containing all parameters necessary for calculating information measure values
            using ``preds_distribution`` and ``target_distribution``.
        special_tokens_map:
            A dictionary mapping tokenizer special tokens into the corresponding integer values.
        verbose:
            An indication of whether a progress bar to be displayed during the embeddings calculation.

    Return:
        A corpus-level InfoLM score.
    """
    preds_distribution = _get_data_distribution(model, preds_dataloader, temperature, idf, special_tokens_map, verbose)
    target_distribution = _get_data_distribution(
        model, target_dataloader, temperature, idf, special_tokens_map, verbose
    )
    # Sort preds and target sentences
    preds_distribution = preds_distribution[preds_dataloader.dataset.sorting_indices]
    target_distribution = target_distribution[target_dataloader.dataset.sorting_indices]
    # Calculate information measure
    return information_measure_cls(preds_distribution, target_distribution)


def infolm(
    preds: Union[str, Sequence[str]],
    target: Union[str, Sequence[str]],
    model_name_or_path: Union[str, os.PathLike] = "bert-base-uncased",
    temperature: float = 0.25,
    information_measure: _ALLOWED_INFORMATION_MEASURE_LITERAL = "kl_divergence",
    idf: bool = True,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None,
    max_length: Optional[int] = None,
    batch_size: int = 64,
    num_threads: int = 0,
    verbose: bool = True,
    return_sentence_level_score: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Calculate `InfoLM`_ [1].

    InfoML corresponds to distance/divergence between predicted and reference sentence discrete distribution using
    one of the following information measures:

        - `KL divergence`_
        - `alpha divergence`_
        - `beta divergence`_
        - `AB divergence`_
        - `Rényi divergence`_
        - L1 distance
        - L2 distance
        - L-infinity distance
        - `Fisher-Rao distance`_

    `InfoLM`_ is a family of untrained embedding-based metrics which addresses some famous flaws of standard
    string-based metrics thanks to the usage of pre-trained masked language models. This family of metrics is mainly
    designed for summarization and data-to-text tasks.

    If you want to use IDF scaling over the whole dataset, please use the class metric.

    The implementation of this metric is fully based HuggingFace `transformers`' package.

    Args:
        preds:
            An iterable of hypothesis corpus.
        target:
            An iterable of reference corpus.
        model_name_or_path:
            A name or a model path used to load `transformers` pretrained model.
        temperature:
            A temperature for calibrating language modelling. For more information, please reference `InfoLM`_ paper.
        information_measure:
            A name of information measure to be used. Please use one of: ['kl_divergence', 'alpha_divergence',
            'beta_divergence', 'ab_divergence', 'renyi_divergence', 'l1_distance', 'l2_distance', 'l_infinity_distance',
            'fisher_rao_distance']
        idf:
            An indication of whether normalization using inverse document frequencies should be used.
        alpha:
            Alpha parameter of the divergence used for alpha, AB and Rényi divergence measures.
        beta:
            Beta parameter of the divergence used for beta and AB divergence measures.
        device:
            A device to be used for calculation.
        max_length:
            A maximum length of input sequences. Sequences longer than `max_length` are to be trimmed.
        batch_size:
            A batch size used for model processing.
        num_threads:
            A number of threads to use for a dataloader.
        verbose:
            An indication of whether a progress bar to be displayed during the embeddings calculation.
        return_sentence_level_score:
            An indication whether a sentence-level InfoLM score to be returned.

    Returns:
        A corpus-level InfoLM score.
        (Optionally) A list of sentence-level InfoLM scores if `return_sentence_level_score=True`.

    Example:
        >>> from torchmetrics.functional.text.infolm import infolm
        >>> preds = ['he read the book because he was interested in world history']
        >>> target = ['he was interested in world history because he read the book']
        >>> infolm(preds, target, model_name_or_path='google/bert_uncased_L-2_H-128_A-2', idf=False)
        tensor(-0.1784)

    References:
        [1] InfoLM: A New Metric to Evaluate Summarization & Data2Text Generation by Pierre Colombo, Chloé Clavel and
        Pablo Piantanida `InfoLM`_
    """
    tokenizer, model = _load_tokenizer_and_model(model_name_or_path, device)
    information_measure_cls = _InformationMeasure(information_measure, alpha, beta)
    max_length = max_length or model.config.max_length
    special_tokens_map = _get_special_tokens_map(tokenizer)

    preds_input_ids, preds_attention_mask, target_input_ids, target_attention_mask = _infolm_update(
        preds, target, tokenizer, max_length
    )
    preds_dataloader = _get_dataloader(preds_input_ids, preds_attention_mask, idf, batch_size, num_threads)
    target_dataloader = _get_dataloader(target_input_ids, target_attention_mask, idf, batch_size, num_threads)

    info_lm_score = _infolm_compute(
        model,
        preds_dataloader,
        target_dataloader,
        temperature,
        idf,
        information_measure_cls,
        special_tokens_map,
        verbose,
    )

    if return_sentence_level_score:
        return info_lm_score.mean(), info_lm_score

    return info_lm_score.mean()

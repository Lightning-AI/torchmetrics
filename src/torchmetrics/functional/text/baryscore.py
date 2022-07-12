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
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torchmetrics.functional.text.helper_embedding_metric import (
    _embedding_metrics_update,
    _get_progress_bar,
    _get_dataloader,
    _input_data_collator,
    _load_tokenizer_and_model,
)

from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase, PretrainedConfig
else:
    PreTrainedModel = PreTrainedTokenizerBase = None  # type: ignore
    __doctest_skip__ = ["baryscore"]

def _check_valid_num_last_layers(config: PretrainedConfig, num_last_layers: int) -> None:
    # Encoder-only or Decoder-only models
    if not config.is_encoder_decoder and num_last_layers > config.num_hidden_layers:
        raise ValueError(
            f"Parameter `num_hidden_layers` is not correctly specified as {num_last_layers=} > "
            f"{config.num_hidden_layers=}."
        )
    # Encoder-Decoder models
    if config.is_encoder_decoder and num_last_layers > config.encoder_layers + config.decoder_layers:
            raise ValueError(
            f"Parameter `num_hidden_layers` is not correctly specified as {num_last_layers=} > "
            f"{config.encoder_layers + config.decoder_layers=}."
        )

def _get_free_support_barycenters(measures_locations: Tensor, tokens_weights: Tensor):
    """
    """
    MAX_ITERATIONS = 1000
    STOPPING_THRESHOLD = 1e-7

    seq_len, model_dim = measures_locations.shape[2], measures_locations[3]
    barycenter_weights = torch.tensor(
        [1 / seq_len] * seq_len, dtype=measures_locations.dtype, device=measures_locations.device
    )

    for sequence_embeddings in measures_locations:
        num_iters = 0
        displacement_square_norm = STOPPING_THRESHOLD + 1.0
        barycenters = torch.zeros(seq_len, model_dim, dtype=measures_locations.dtype, device=measures_locations.device)

        while displacement_square_norm > STOPPING_THRESHOLD and num_iters < MAX_ITERATIONS:
            transport_cost_sum = torch.zeros(
                seq_len, model_dim, dtype=measures_locations.dtype, device=measures_locations.device
            )



def _get_batch_wasserstein_barycenters(
    model: PreTrainedModel, batch: Dict[str, Tensor], num_last_layers: int, idf: bool
) -> Tensor:
    """
    Args:
        model:
            Initialized model from HuggingFace's `transformers package.
        batch:
            An input batch dictionary containing ``input_ids`` and ``attention_mask``.
        num_last_layers:
            A number of how many of the last model layers should be used for BaryScore calculation.
        idf:
            An indication of whether normalization using inverse document frequencies should be used.
    """
    hidden_states = model(batch["input_ids"], batch["attention_mask"], output_hidden_states=True).hidden_states
    hidden_states = torch.cat([layer_states.unsqueeze(0) for layer_states in hidden_states[-num_last_layers:]], dim=0)
    # [num_last_layers, batch_size, seq_len, model_dim]
    hidden_states /= hidden_states.norm(dim=-1).unsqueeze(-1)
    hidden_states = torch.einsum("lbsd, bs -> lbsd", hidden_states, batch["attention_mask"])
    hidden_states.type(torch.float64)

    if idf:
        token_weights = batch["input_ids_idf"] / batch["input_ids_idf"].sum(-1).unsqueeze(-1)
    else:
        token_weights = batch["attention_mask"] / batch["attention_mask"].sum(-1).unsqueeze(-1)
    
    wasserstein_barycenters = _get_free_support_barycenters(hidden_states, token_weights.type(hidden_states.dtype))
    return wasserstein_barycenters.cpu()


@torch.no_grad()
def _get_wasserstein_barycenters(
    model: PreTrainedModel, dataloader: DataLoader, num_last_layers: int, idf: bool, verbose: bool
) -> Tensor:
    """
    Args:
        model:
            Initialized model from HuggingFace's `transformers package.
        dataloader:
            An instance of `torch.utils.data.DataLoader` used for iterating over examples.
        num_last_layers:
            A number of how many of the last model layers should be used for BaryScore calculation.
        idf:
            An indication of whether normalization using inverse document frequencies should be used.
        verbose:
            An indication of whether a progress bar to be displayed during the embeddings calculation.

    Return:
        .
    """
    device = model.device
    wasserstein_barycenters: List[Tensor] = []

    for batch in _get_progress_bar(dataloader, verbose):
        batch = _input_data_collator(batch, device)
        wasserstein_barycenters.append(_get_batch_wasserstein_barycenters((model, batch, num_last_layers, idf)))

    return torch.cat(wasserstein_barycenters, dim=0)


def _baryscore_update(
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
    return _embedding_metrics_update(preds, target, tokenizer, max_length)

def _baryscore_compute(
    model: PreTrainedModel,
    preds_dataloader: DataLoader,
    target_dataloader: DataLoader,
    num_last_layers: int,
    idf: bool,
    verbose: bool = True,
):
    """Calculate BaryScore using the pre-trained language model.

    Args:
        model:
            Initialized model from HuggingFace's `transformers package.
        preds_dataloader:
            Loader iterating over tokenizer predicted sentences.
        target_dataloader:
            Loader iterating over tokenizer reference sentences.
        num_last_layers:
            A number of how many of the last model layers should be used for BaryScore calculation.
        idf:
            An indication of whether normalization using inverse document frequencies should be used.
        verbose:
            An indication of whether a progress bar to be displayed during the embeddings calculation.

    Return:
        A sentence-level BaryScore.
    """
    preds_wasserstein_barycenters = _get_wasserstein_barycenters(model, preds_dataloader, num_last_layers, idf, verbose)
    target_wasserstein_barycenters = _get_wasserstein_barycenters(model, target_dataloader, num_last_layers, idf, verbose)
    # Sort preds and target sentences
    preds_wasserstein_barycenters = preds_wasserstein_barycenters[preds_dataloader.dataset.sorting_indices]
    target_wasserstein_barycenters = target_wasserstein_barycenters[target_dataloader.dataset.sorting_indices]
    

def baryscore(
    preds: Union[str, Sequence[str]],
    target: Union[str, Sequence[str]],
    model_name_or_path: Union[str, os.PathLike] = "bert-base-uncased",
    num_last_layers: int = 1,
    idf: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    max_length: Optional[int] = None,
    batch_size: int = 64,
    num_threads: int = 0,
    verbose: bool = True,
    return_sentence_level_score: bool = False,
):
    """
    Args:
        preds:
            An iterable of hypothesis corpus.
        target:
            An iterable of reference corpus.
        model_name_or_path:
            A name or a model path used to load `transformers` pretrained model.
        num_last_layers:
            A number of how many of the last model layers should be used for BaryScore calculation.
        idf:
            An indication of whether normalization using inverse document frequencies should be used.
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

    Example:
        >>> from torchmetrics.functional.text.baryscore import baryscore
        >>> preds = ['he read the book because he was interested in world history']
        >>> target = ['he was interested in world history because he read the book']
    References:
    """
    tokenizer, model = _load_tokenizer_and_model(model_name_or_path, device)
    _check_valid_num_last_layers(model.config, num_last_layers)
    max_length = max_length or model.config.max_length

    preds_input_ids, preds_attention_mask, target_input_ids, target_attention_mask = _baryscore_update(
        preds, target, tokenizer, max_length
    )
    preds_dataloader = _get_dataloader(preds_input_ids, preds_attention_mask, idf, batch_size, num_threads)
    target_dataloader = _get_dataloader(target_input_ids, target_attention_mask, idf, batch_size, num_threads)

    bary_score = _baryscore_compute(model, preds_dataloader, target_dataloader, num_last_layers, idf, verbose)

    if return_sentence_level_score:
        return bary_score.mean(), bary_score

    return bary_score.mean()
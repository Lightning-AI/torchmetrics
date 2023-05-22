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
import csv
import os
import urllib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from torchmetrics.functional.text.helper_embedding_metric import (
    TextDataset,
    TokenizedDataset,
    _check_shape_of_model_output,
    _get_progress_bar,
    _input_data_collator,
    _output_data_collator,
    _process_attention_mask_for_special_tokens,
)
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_AVAILABLE

# Default model recommended in the original implementation.
_DEFAULT_MODEL = "roberta-large"

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoModel, AutoTokenizer

    def _download_model() -> None:
        """Download intensive operations."""
        AutoTokenizer.from_pretrained(_DEFAULT_MODEL)
        AutoModel.from_pretrained(_DEFAULT_MODEL)

    if _SKIP_SLOW_DOCTEST and not _try_proceed_with_timeout(_download_model):
        __doctest_skip__ = ["bert_score"]
else:
    __doctest_skip__ = ["bert_score"]


def _get_embeddings_and_idf_scale(
    dataloader: DataLoader,
    target_len: int,
    model: Module,
    device: Optional[Union[str, torch.device]] = None,
    num_layers: Optional[int] = None,
    all_layers: bool = False,
    idf: bool = False,
    verbose: bool = False,
    user_forward_fn: Optional[Callable[[Module, Dict[str, Tensor]], Tensor]] = None,
) -> Tuple[Tensor, Tensor]:
    """Calculate sentence embeddings and the inverse-document-frequency scaling factor.

    Args:
        dataloader: dataloader instance.
        target_len: A length of the longest sequence in the data. Used for padding the model output.
        model: BERT model.
        device: A device to be used for calculation.
        num_layers: The layer of representation to use.
        all_layers: An indication whether representation from all model layers should be used for BERTScore.
        idf: An Indication whether normalization using inverse document frequencies should be used.
        verbose: An indication of whether a progress bar to be displayed during the embeddings' calculation.
        user_forward_fn:
            A user's own forward function used in a combination with ``user_model``. This function must
            take ``user_model`` and a python dictionary of containing ``"input_ids"`` and ``"attention_mask"``
            represented by :class:`~torch.Tensor` as an input and return the model's output represented by the single
            :class:`~torch.Tensor`.

    Return:
        A tuple of :class:`~torch.Tensor`s containing the model's embeddings and the normalized tokens IDF.
        When ``idf = False``, tokens IDF is not calculated, and a matrix of mean weights is returned instead.
        For a single sentence, ``mean_weight = 1/seq_len``, where ``seq_len`` is a sum over the corresponding
        ``attention_mask``.

    Raises:
        ValueError:
            If ``all_layers = True`` and a model, which is not from the ``transformers`` package, is used.
    """
    embeddings_list: List[Tensor] = []
    idf_scale_list: List[Tensor] = []
    for batch in _get_progress_bar(dataloader, verbose):
        with torch.no_grad():
            batch = _input_data_collator(batch, device)
            # Output shape: batch_size x num_layers OR 1 x sequence_length x bert_dim
            if not all_layers:
                if not user_forward_fn:
                    out = model(batch["input_ids"], batch["attention_mask"], output_hidden_states=True)
                    out = out.hidden_states[num_layers if num_layers is not None else -1]
                else:
                    out = user_forward_fn(model, batch)
                    _check_shape_of_model_output(out, batch["input_ids"])
                out = out.unsqueeze(1)
            else:
                if user_forward_fn:
                    raise ValueError(
                        "The option `all_layers=True` can be used only with default `transformers` models."
                    )
                out = model(batch["input_ids"], batch["attention_mask"], output_hidden_states=True)
                out = torch.cat([o.unsqueeze(1) for o in out.hidden_states], dim=1)

        out /= out.norm(dim=-1).unsqueeze(-1)  # normalize embeddings
        out, attention_mask = _output_data_collator(out, batch["attention_mask"], target_len)
        processed_attention_mask = _process_attention_mask_for_special_tokens(attention_mask)
        # Multiply embeddings with attention_mask (b=batch_size, l=num_layers, s=seq_len, d=emb_dim)
        out = torch.einsum("blsd, bs -> blsd", out, processed_attention_mask)
        embeddings_list.append(out.cpu())

        # Calculate weighted (w.r.t. sentence length) input_ids IDF matrix
        input_ids_idf = (
            batch["input_ids_idf"] * processed_attention_mask if idf else processed_attention_mask.type(out.dtype)
        )
        input_ids_idf /= input_ids_idf.sum(-1, keepdim=True)
        idf_scale_list.append(input_ids_idf.cpu())

    embeddings = torch.cat(embeddings_list)
    idf_scale = torch.cat(idf_scale_list)

    return embeddings, idf_scale


def _get_scaled_precision_or_recall(cos_sim: Tensor, metric: str, idf_scale: Tensor) -> Tensor:
    """Calculate precision or recall, transpose it and scale it with idf_scale factor."""
    dim = 3 if metric == "precision" else 2
    res = cos_sim.max(dim=dim).values
    res = torch.einsum("bls, bs -> bls", res, idf_scale).sum(-1)
    # We transpose the results and squeeze if possible to match the format of the original BERTScore implementation
    return res.transpose(0, 1).squeeze()


def _get_precision_recall_f1(
    preds_embeddings: Tensor, target_embeddings: Tensor, preds_idf_scale: Tensor, target_idf_scale: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Calculate precision, recall and F1 score over candidate and reference sentences.

    Args:
        preds_embeddings: Embeddings of candidate sentences.
        target_embeddings: Embeddings of reference sentences.
        preds_idf_scale: An IDF scale factor for candidate sentences.
        target_idf_scale: An IDF scale factor for reference sentences.

    Return:
        Tensors containing precision, recall and F1 score, respectively.
    """
    # Dimensions: b = batch_size, l = num_layers, p = predictions_seq_len, r = references_seq_len, d = bert_dim
    cos_sim = torch.einsum("blpd, blrd -> blpr", preds_embeddings, target_embeddings)
    # Final metrics shape = (batch_size * num_layers | batch_size)
    precision = _get_scaled_precision_or_recall(cos_sim, "precision", preds_idf_scale)
    recall = _get_scaled_precision_or_recall(cos_sim, "recall", target_idf_scale)

    f1_score = 2 * precision * recall / (precision + recall)
    f1_score = f1_score.masked_fill(torch.isnan(f1_score), 0.0)

    return precision, recall, f1_score


def _get_hash(model_name_or_path: Optional[str] = None, num_layers: Optional[int] = None, idf: bool = False) -> str:
    """Compute `BERT_score`_ (copied and adjusted)."""
    return f"{model_name_or_path}_L{num_layers}{'_idf' if idf else '_no-idf'}"


def _read_csv_from_local_file(baseline_path: str) -> Tensor:
    """Read baseline from csv file from the local file.

    This method implemented to avoid `pandas` dependency.
    """
    with open(baseline_path) as fname:
        csv_file = csv.reader(fname)
        baseline_list = [[float(item) for item in row] for idx, row in enumerate(csv_file) if idx > 0]
    return torch.tensor(baseline_list)[:, 1:]


def _read_csv_from_url(baseline_url: str) -> Tensor:
    """Read baseline from csv file from URL.

    This method is implemented to avoid `pandas` dependency.
    """
    with urllib.request.urlopen(baseline_url) as http_request:
        baseline_list = [
            [float(item) for item in row.strip().decode("utf-8").split(",")]
            for idx, row in enumerate(http_request)
            if idx > 0
        ]
        return torch.tensor(baseline_list)[:, 1:]


def _load_baseline(
    lang: str = "en",
    model_name_or_path: Optional[str] = None,
    baseline_path: Optional[str] = None,
    baseline_url: Optional[str] = None,
) -> Optional[Tensor]:
    """Load a CSV file with the baseline values used for rescaling."""
    if baseline_path:
        baseline: Optional[Tensor] = _read_csv_from_local_file(baseline_path)
    elif baseline_url:
        baseline = _read_csv_from_url(baseline_url)
    # Read default baseline from the original `bert-score` package https://github.com/Tiiiger/bert_score
    elif lang and model_name_or_path:
        url_base = "https://raw.githubusercontent.com/Tiiiger/bert_score/master/bert_score/rescale_baseline"
        baseline_url = f"{url_base}/{lang}/{model_name_or_path}.tsv"
        baseline = _read_csv_from_url(baseline_url)
    else:
        rank_zero_warn("Baseline was not successfully loaded. No baseline is going to be used.")
        return None

    return baseline


def _rescale_metrics_with_baseline(
    precision: Tensor,
    recall: Tensor,
    f1_score: Tensor,
    baseline: Tensor,
    num_layers: Optional[int] = None,
    all_layers: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rescale the computed metrics with the pre-computed baseline."""
    if num_layers is None and all_layers is False:
        num_layers = -1
    all_metrics = torch.stack([precision, recall, f1_score], dim=-1)
    baseline_scale = baseline.unsqueeze(1) if all_layers else baseline[num_layers]
    all_metrics = (all_metrics - baseline_scale) / (1 - baseline_scale)

    return all_metrics[..., 0], all_metrics[..., 1], all_metrics[..., 2]


def bert_score(
    preds: Union[str, Sequence[str], Dict[str, Tensor]],
    target: Union[str, Sequence[str], Dict[str, Tensor]],
    model_name_or_path: Optional[str] = None,
    num_layers: Optional[int] = None,
    all_layers: bool = False,
    model: Optional[Module] = None,
    user_tokenizer: Any = None,
    user_forward_fn: Optional[Callable[[Module, Dict[str, Tensor]], Tensor]] = None,
    verbose: bool = False,
    idf: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    max_length: int = 512,
    batch_size: int = 64,
    num_threads: int = 0,
    return_hash: bool = False,
    lang: str = "en",
    rescale_with_baseline: bool = False,
    baseline_path: Optional[str] = None,
    baseline_url: Optional[str] = None,
) -> Dict[str, Union[Tensor, List[float], str]]:
    """`Bert_score Evaluating Text Generation`_ for text similirity matching.

    This metric leverages the pre-trained contextual embeddings from BERT and matches words in candidate and reference
    sentences by cosine similarity. It has been shown to correlate with human judgment on sentence-level and
    system-level evaluation. Moreover, BERTScore computes precision, recall, and F1 measure, which can be useful for
    evaluating different language generation tasks.

    This implemenation follows the original implementation from `BERT_score`_.

    Args:
        preds: Either an iterable of predicted sentences or a ``Dict[input_ids, attention_mask]``.
        target: Either an iterable of target sentences or a  ``Dict[input_ids, attention_mask]``.
        model_name_or_path: A name or a model path used to load ``transformers`` pretrained model.
        num_layers: A layer of representation to use.
        all_layers:
            An indication of whether the representation from all model's layers should be used.
            If ``all_layers = True``, the argument ``num_layers`` is ignored.
        model: A user's own model.
        user_tokenizer:
            A user's own tokenizer used with the own model. This must be an instance with the ``__call__`` method.
            This method must take an iterable of sentences (``List[str]``) and must return a python dictionary
            containing ``"input_ids"`` and ``"attention_mask"`` represented by :class:`~torch.Tensor`.
            It is up to the user's model of whether ``"input_ids"`` is a :class:`~torch.Tensor` of input ids
            or embedding vectors. his tokenizer must prepend an equivalent of ``[CLS]`` token and append an equivalent
            of ``[SEP]`` token as `transformers` tokenizer does.
        user_forward_fn:
            A user's own forward function used in a combination with ``user_model``.
            This function must take ``user_model`` and a python dictionary of containing ``"input_ids"``
            and ``"attention_mask"`` represented by :class:`~torch.Tensor` as an input and return the model's output
            represented by the single :class:`~torch.Tensor`.
        verbose: An indication of whether a progress bar to be displayed during the embeddings' calculation.
        idf: An indication of whether normalization using inverse document frequencies should be used.
        device: A device to be used for calculation.
        max_length: A maximum length of input sequences. Sequences longer than ``max_length`` are to be trimmed.
        batch_size: A batch size used for model processing.
        num_threads: A number of threads to use for a dataloader.
        return_hash: An indication of whether the correspodning ``hash_code`` should be returned.
        lang: A language of input sentences. It is used when the scores are rescaled with a baseline.
        rescale_with_baseline:
            An indication of whether bertscore should be rescaled with a pre-computed baseline.
            When a pretrained model from ``transformers`` model is used, the corresponding baseline is downloaded
            from the original ``bert-score`` package from `BERT_score`_ if available.
            In other cases, please specify a path to the baseline csv/tsv file, which must follow the formatting
            of the files from `BERT_score`_
        baseline_path: A path to the user's own local csv/tsv file with the baseline scale.
        baseline_url: A url path to the user's own  csv/tsv file with the baseline scale.

    Returns:
        Python dictionary containing the keys ``precision``, ``recall`` and ``f1`` with corresponding values.

    Raises:
        ValueError:
            If ``len(preds) != len(target)``.
        ModuleNotFoundError:
            If `tqdm` package is required and not installed.
        ModuleNotFoundError:
            If ``transformers`` package is required and not installed.
        ValueError:
            If ``num_layer`` is larger than the number of the model layers.
        ValueError:
            If invalid input is provided.

    Example:
        >>> from pprint import pprint
        >>> from torchmetrics.functional.text.bert import bert_score
        >>> preds = ["hello there", "general kenobi"]
        >>> target = ["hello there", "master kenobi"]
        >>> pprint(bert_score(preds, target))
        {'f1': tensor([1.0000, 0.9961]), 'precision': tensor([1.0000, 0.9961]), 'recall': tensor([1.0000, 0.9961])}
    """
    if len(preds) != len(target):
        raise ValueError("Number of predicted and reference sententes must be the same!")
    if not isinstance(preds, (str, list, dict)):  # dict for BERTScore class compute call
        preds = list(preds)
    if not isinstance(target, (str, list, dict)):  # dict for BERTScore class compute call
        target = list(target)

    if verbose and (not _TQDM_AVAILABLE):
        raise ModuleNotFoundError(
            "An argument `verbose = True` requires `tqdm` package be installed. Install with `pip install tqdm`."
        )

    if model is None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ModuleNotFoundError(
                "`bert_score` metric with default models requires `transformers` package be installed."
                " Either install with `pip install transformers>=4.0` or `pip install torchmetrics[text]`."
            )
        if model_name_or_path is None:
            rank_zero_warn(
                "The argument `model_name_or_path` was not specified while it is required when default"
                " `transformers` model are used."
                f"It is, therefore, used the default recommended model - {_DEFAULT_MODEL}."
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path or _DEFAULT_MODEL)
        model = AutoModel.from_pretrained(model_name_or_path or _DEFAULT_MODEL)
    else:
        tokenizer = user_tokenizer
    model.eval()
    model.to(device)

    try:
        if num_layers and num_layers > model.config.num_hidden_layers:  # type: ignore
            raise ValueError(
                f"num_layers={num_layers} is forbidden for {model_name_or_path}."  # type: ignore
                f" Please use num_layers <= {model.config.num_hidden_layers}"
            )
    except AttributeError:
        rank_zero_warn("It was not possible to retrieve the parameter `num_layers` from the model specification.")

    _are_empty_lists = all(isinstance(text, list) and len(text) == 0 for text in (preds, target))
    _are_valid_lists = all(
        isinstance(text, list) and len(text) > 0 and isinstance(text[0], str) for text in (preds, target)
    )
    _are_valid_tensors = all(
        isinstance(text, dict) and isinstance(text["input_ids"], Tensor) for text in (preds, target)
    )
    if _are_empty_lists:
        rank_zero_warn("Predictions and references are empty.")
        output_dict: Dict[str, Union[Tensor, List[float], str]] = {
            "precision": [0.0],
            "recall": [0.0],
            "f1": [0.0],
        }
        if return_hash:
            output_dict.update({"hash": _get_hash(model_name_or_path, num_layers, idf)})
        return output_dict

    # Load baselines if needed
    baseline = _load_baseline(lang, model_name_or_path, baseline_path, baseline_url) if rescale_with_baseline else None

    # We ignore mypy typing below as the proper typing is ensured by conditions above, only mypy cannot infer that.
    if _are_valid_lists:
        target_dataset = TextDataset(target, tokenizer, max_length, idf=idf)  # type: ignore
        preds_dataset = TextDataset(
            preds,  # type: ignore
            tokenizer,
            max_length,
            idf=idf,
            tokens_idf=target_dataset.tokens_idf,
        )
    elif _are_valid_tensors:
        target_dataset = TokenizedDataset(**target, idf=idf)  # type: ignore
        preds_dataset = TokenizedDataset(**preds, idf=idf, tokens_idf=target_dataset.tokens_idf)  # type: ignore
    else:
        raise ValueError("Invalid input provided.")

    target_loader = DataLoader(target_dataset, batch_size=batch_size, num_workers=num_threads)
    preds_loader = DataLoader(preds_dataset, batch_size=batch_size, num_workers=num_threads)

    target_embeddings, target_idf_scale = _get_embeddings_and_idf_scale(
        target_loader, target_dataset.max_length, model, device, num_layers, all_layers, idf, verbose, user_forward_fn
    )
    preds_embeddings, preds_idf_scale = _get_embeddings_and_idf_scale(
        preds_loader, preds_dataset.max_length, model, device, num_layers, all_layers, idf, verbose, user_forward_fn
    )

    precision, recall, f1_score = _get_precision_recall_f1(
        preds_embeddings, target_embeddings, preds_idf_scale, target_idf_scale
    )
    # Sort predictions
    if len(precision.shape) == 1:  # i.e. when all_layers = False
        precision = precision[preds_loader.dataset.sorting_indices]
        recall = recall[preds_loader.dataset.sorting_indices]
        f1_score = f1_score[preds_loader.dataset.sorting_indices]
    elif len(precision.shape) == 2:  # i.e. when all_layers = True
        precision = precision[:, preds_loader.dataset.sorting_indices]
        recall = recall[:, preds_loader.dataset.sorting_indices]
        f1_score = f1_score[:, preds_loader.dataset.sorting_indices]

    if baseline is not None:
        precision, recall, f1_score = _rescale_metrics_with_baseline(
            precision, recall, f1_score, baseline, num_layers, all_layers
        )

    output_dict = {
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
    }
    if return_hash:
        output_dict.update({"hash": _get_hash(model_name_or_path, num_layers, idf)})
    return output_dict

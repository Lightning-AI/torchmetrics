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
import csv
import math
import urllib
import warnings
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoModel, AutoTokenizer

if _TQDM_AVAILABLE:
    import tqdm


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
        trunction:
            An indication of whether tokenized sequences should be padded only to the length of the longest sequence.
        sort_according_to_length:
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


def _process_attention_mask_for_special_tokens(attention_mask: Tensor) -> Tensor:
    """Process attention mask to be zero for special [CLS] and [SEP] tokens as they're not included in a
    calculation for BERT score.

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


def _sort_data_according_length(input_ids: Tensor, attention_mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Sort tokenized sentence from the shortest to the longest one."""
    sorted_indices = attention_mask.sum(1).argsort()
    input_ids = input_ids[sorted_indices]
    attention_mask = attention_mask[sorted_indices]
    return input_ids, attention_mask


def _input_data_collator(
    batch: Dict[str, Tensor], device: Optional[Union[str, torch.device]] = None
) -> Dict[str, Tensor]:
    """Helper function that trims model inputs to the longest sequence within the batch and put the input on the
    proper device."""
    max_len = int(batch["attention_mask"].sum(1).max().item())
    input_ids = batch["input_ids"][:, :max_len].to(device)
    attention_mask = batch["attention_mask"][:, :max_len].to(device)
    batch.update({"input_ids": input_ids, "attention_mask": attention_mask})
    return batch


def _output_data_collator(model_output: Tensor, attention_mask: Tensor, target_len: int) -> Tuple[Tensor, Tensor]:
    """Helper function that pads the model output and attention mask to the target length."""
    zeros_shape = list(model_output.shape)
    zeros_shape[2] = target_len - zeros_shape[2]
    model_output = torch.cat(
        [model_output, torch.zeros(zeros_shape, dtype=model_output.dtype).to(model_output.device)], dim=2
    )
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.zeros(zeros_shape[0], zeros_shape[2], dtype=attention_mask.dtype).to(attention_mask.device),
        ],
        dim=1,
    )
    return model_output, attention_mask


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
            own_tokenizer:
                An indication of whether a non-default user's own tokenizer is used.
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


class TokenizedDataset(TextDataset):
    """The child class of `TextDataset` class used with already tokenized data."""

    def __init__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        idf: bool = False,
        tokens_idf: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Args:
            input_ids:
                Input ids (`torch.Tensor`).
            attention_mask:
                Attention mask (`torch.Tensor`).
            idf:
                An indication of whether calculate token inverse document frequencies to weight the model embeddings.
            tokens_idf:
                Inverse document frequencies (these should be calculated on reference sentences).
        """
        self.text = dict(zip(["input_ids", "attention_mask"], _sort_data_according_length(input_ids, attention_mask)))
        self.text = _input_data_collator(self.text)
        self.num_sentences = len(self.text["input_ids"])
        self.max_length = self.text["input_ids"].shape[1]
        self.idf = idf
        self.tokens_idf = {}
        if idf:
            self.tokens_idf = tokens_idf if tokens_idf is not None else self._get_tokens_idf()


def _get_progress_bar(dataloader: DataLoader, verbose: bool = False) -> Union[DataLoader, tqdm.auto.tqdm]:
    """Helper function returning either the dataloader itself when `verbose = False`, or it wraps the dataloader with
    `tqdm.auto.tqdm`, when `verbose = True` to display a progress bar during the embbeddings calculation."""
    return tqdm.auto.tqdm(dataloader) if verbose else dataloader


def _check_shape_of_model_output(output: Tensor, input_ids: Tensor) -> None:
    """Check if the shape of the user's own model output."""
    bs, seq_len = input_ids.shape[:2]
    invalid_out_shape = len(output.shape) != 3 or output.shape[0] != bs or output.shape[1] != seq_len
    if invalid_out_shape:
        raise ValueError(
            "The model output must be `torch.Tensor` of a shape `[batch_size, seq_len, model_dim]` "
            f"i.e. [{bs}, {seq_len}. , `model_dim`], but got {output.shape}."
        )


def _get_embeddings_and_idf_scale(
    dataloader: DataLoader,
    target_len: int,
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    num_layers: Optional[int] = None,
    all_layers: bool = False,
    idf: bool = False,
    verbose: bool = False,
    user_forward_fn: Callable[[torch.nn.Module, Dict[str, Tensor]], Tensor] = None,
) -> Tuple[Tensor, Tensor]:
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
        verbose:
            An indication of whether a progress bar to be displayed during the embeddings calculation.
        user_forward_fn:
            A user's own forward function used in a combination with `user_model`. This function must take `user_model`
            and a python dictionary of containing `"input_ids"` and `"attention_mask"` represented by `torch.Tensor`
            as an input and return the model's output represented by the single `torch.Tensor`.

    Return:
        A tuple of torch.Tensors containing the model's embeddings and the normalized tokens IDF.
        When `idf = False`, tokens IDF is not calculated, and a matrix of mean weights is returned instead.
        For a single sentence, `mean_weight = 1/seq_len`, where `seq_len` is a sum over the corresponding
        `attention_mask`.

    Raises:
        ValueError:
            If `all_layers = True` and a model, which is not from the `transformers` package, is used.
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
        idf_scale_list.append(input_ids_idf)

    embeddings = torch.cat(embeddings_list)
    idf_scale = torch.cat(idf_scale_list)

    return embeddings, idf_scale


def _get_scaled_precision_or_recall(cos_sim: Tensor, metric: str, idf_scale: Tensor) -> Tensor:
    """Helper function that calculates precision or recall, transpose it and scale it with idf_scale factor."""
    dim = 3 if metric == "precision" else 2
    res = cos_sim.max(dim=dim).values
    res = torch.einsum("bls, bs -> bls", res, idf_scale).sum(-1)
    # We transpose the results and squeeze if possible to match the format of the original BERTScore implementation
    res = res.transpose(0, 1).squeeze()
    return res


def _get_precision_recall_f1(
    pred_embeddings: Tensor, ref_embeddings: Tensor, pred_idf_scale: Tensor, ref_idf_scale: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
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


def _get_hash(model_name_or_path: Optional[str] = None, num_layers: Optional[int] = None, idf: bool = False) -> str:
    """Copied from https://github.com/Tiiiger/bert_score/blob/master/bert_score/utils.py and adjusted."""
    msg = f"{model_name_or_path}_L{num_layers}{'_idf' if idf else '_no-idf'}"
    return msg


def _read_csv_from_local_file(baseline_path: str) -> Tensor:
    """Helper function which reads baseline the csv file from the local file.

    This method implemented to avoid `pandas` dependency.
    """
    with open(baseline_path) as fname:
        csv_file = csv.reader(fname)
        baseline_list = [[float(item) for item in row] for idx, row in enumerate(csv_file) if idx > 0]
    baseline = torch.tensor(baseline_list)[:, 1:]
    return baseline


def _read_csv_from_url(baseline_url: str) -> Tensor:
    """Helper function which reads the baseline csv file from URL.

    This method is implemented to avoid `pandas` dependency.
    """
    with urllib.request.urlopen(baseline_url) as http_request:  # type: ignore
        baseline_list = [
            [float(item) for item in row.strip().decode("utf-8").split(",")]
            for idx, row in enumerate(http_request)
            if idx > 0
        ]
        baseline = torch.tensor(baseline_list)[:, 1:]
    return baseline


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
        _URL_BASE = "https://raw.githubusercontent.com/Tiiiger/bert_score/master/bert_score/rescale_baseline"
        baseline_url = f"{_URL_BASE}/{lang}/{model_name_or_path}.tsv"
        baseline = _read_csv_from_url(baseline_url)
    else:
        baseline = None
        warnings.warn("Baseline was not successfully loaded. No baseline is going to be used.")

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
    predictions: Union[List[str], Dict[str, Tensor]],
    references: Union[List[str], Dict[str, Tensor]],
    model_name_or_path: Optional[str] = None,
    num_layers: Optional[int] = None,
    all_layers: bool = False,
    model: Optional[torch.nn.Module] = None,
    user_tokenizer: Any = None,
    user_forward_fn: Callable[[torch.nn.Module, Dict[str, Tensor]], Tensor] = None,
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
) -> Dict[str, Union[List[float], str]]:
    """`BERTScore <https://arxiv.org/abs/1904.09675>`_ leverages the pre-trained contextual embeddings from BERT
    and matches words in candidate and reference sentences by cosine similarity. It has been shown to correlate
    with human judgment on sentence-level and system-level evaluation. Moreover, BERTScore computes precision,
    recall, and F1 measure, which can be useful for evaluating different language generation tasks. assert
    len(predictions) == len(references), "Number of predicted and reference sententes must be the same!".

    This implemenation follows the original implementation from https://github.com/Tiiiger/bert_score.

    Args:
        predictions:
            Either an iterable of predicted sentences or a `Dict[str, torch.Tensor]` containing `input_ids` and
            `attention_mask` `torch.Tensor`.
        references:
            Either an iterable of target sentences or a `Dict[str, torch.Tensor]` containing `input_ids` and
            `attention_mask` `torch.Tensor`.
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
            An indication of whether normalization using inverse document frequencies should be used.
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
            A language of input sentences. It is used when the scores are rescaled with a baseline.
        rescale_with_baseline:
            An indication of whether bertscore should be rescaled with a pre-computed baseline.
            When a pretrained model from `transformers` model is used, the corresponding baseline is downloaded
            from the original `bert-score` package from https://github.com/Tiiiger/bert_score if available.
            In other cases, please specify a path to the baseline csv/tsv file, which must follow the formatting
            of the files from https://github.com/Tiiiger/bert_score.
        baseline_path:
            A path to the user's own local csv/tsv file with the baseline scale.
        baseline_url:
            A url path to the user's own  csv/tsv file with the baseline scale.

    Returns:
        Python dictionary containing the keys `precision`, `recall` and `f1` with corresponding values.

    Raises:
        ValueError:
            If `len(predictions) != len(references)`.
        ValueError:
            If `tqdm` package is required and not installed.
        ValueError:
            If `transformers` package is required and not installed.
        ValueError:
            If `num_layer` is larger than the number of the model layers.
        ValueError:
            If invalid input is provided.

    Example:
        >>> predictions = ["hello there", "general kenobi"]
        >>> references = ["hello there", "master kenobi"]
        >>> bert_score(predictions=predictions, references=references, lang="en")  # doctest: +SKIP
        {'precision': [0.99..., 0.99...],
         'recall': [0.99..., 0.99...],
         'f1': [0.99..., 0.99...]}
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predicted and reference sententes must be the same!")

    if verbose and (not _TQDM_AVAILABLE):
        raise ValueError(
            "An argument `verbose = True` requires `tqdm` package be installed. Install with `pip install tqdm`."
        )

    if model is None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ValueError(
                "`bert_score` metric with default models requires `transformers` package be installed. "
                "Either install with `pip install transformers>=4.0` or `pip install torchmetrics[text]`"
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
    else:
        tokenizer = user_tokenizer
    model.eval()
    model.to(device)

    try:
        if num_layers and num_layers > model.config.num_hidden_layers:  # type: ignore
            raise ValueError(
                f"num_layers={num_layers} is forbidden for {model_name_or_path}. "  # type: ignore
                f"Please use num_layers <= {model.config.num_hidden_layers}"  # type: ignore
            )
    except AttributeError:
        warnings.warn("It was not possible to retrieve the parameter `num_layers` from the model specification.")

    _are_empty_lists = all(isinstance(text, list) and len(text) == 0 for text in (predictions, references))
    _are_valid_lists = all(
        isinstance(text, list) and len(text) > 0 and isinstance(text[0], str) for text in (predictions, references)
    )
    _are_valid_tensors = all(
        isinstance(text, dict) and isinstance(text["input_ids"], Tensor) for text in (predictions, references)
    )
    if _are_empty_lists:
        warnings.warn("Predictions and references are empty.")
        output_dict: Dict[str, Union[List[float], str]] = {
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
        ref_dataset = TextDataset(references, tokenizer, max_length, idf=idf)  # type: ignore
        pred_dataset = TextDataset(
            predictions,  # type: ignore
            tokenizer,
            max_length,
            idf=idf,
            tokens_idf=ref_dataset.tokens_idf,
        )
    elif _are_valid_tensors:
        ref_dataset = TokenizedDataset(**references, idf=idf)  # type: ignore
        pred_dataset = TokenizedDataset(**predictions, idf=idf, tokens_idf=ref_dataset.tokens_idf)  # type: ignore
    else:
        raise ValueError("Invalid input provided.")

    ref_loader = DataLoader(ref_dataset, batch_size=batch_size, num_workers=num_threads)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, num_workers=num_threads)

    ref_embeddings, ref_idf_scale = _get_embeddings_and_idf_scale(
        ref_loader, ref_dataset.max_length, model, device, num_layers, all_layers, idf, verbose, user_forward_fn
    )
    pred_embeddings, pred_idf_scale = _get_embeddings_and_idf_scale(
        pred_loader, pred_dataset.max_length, model, device, num_layers, all_layers, idf, verbose, user_forward_fn
    )

    precision, recall, f1_score = _get_precision_recall_f1(
        pred_embeddings, ref_embeddings, pred_idf_scale, ref_idf_scale
    )

    if baseline is not None:
        precision, recall, f1_score = _rescale_metrics_with_baseline(
            precision, recall, f1_score, baseline, num_layers, all_layers
        )

    output_dict = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1_score.tolist(),
    }
    if return_hash:
        output_dict.update({"hash": _get_hash(model_name_or_path, num_layers, idf)})
    return output_dict

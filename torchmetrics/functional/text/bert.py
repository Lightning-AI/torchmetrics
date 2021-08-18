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
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.utilities.imports import _BERTSCORE_AVAILABLE, _TRANSFORMERS_AVAILABLE

if _BERTSCORE_AVAILABLE:
    from bert_score import BERTScorer, get_hash, lang2model, model2layers

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoTokenizer, AutoModel


def _preprocess_text(text: List[str], tokenizer: Any, max_length: int = 512) -> Dict[str, torch.Tensor]:
    """Default text pre-processing function using `transformers` `AutoTokenizer` instance.

    Args:
        text: An iterable of sentences.
        tokenizer: `AutoTokenizer` instance from `transformers` package.
        max_length: A maximum sequence length.

    Return:
        A dictionary of tokenized sentences including input_ids and attention_mask.
    """
    return tokenizer(
        text,
        add_special_tokens=False,  # we don't want to add [CLS], [SEP] either other special tokens
        padding=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )


class TextDataset(Dataset):

    def __init__(
        self,
        text: List[str],
        tokenizer: Any,
        max_length: int = 512,
        preprocess_text_fn: Callable[[List[str], Any, int], Dict[str, torch.Tensor]] = _preprocess_text,
        tokens_idf: Optional[Dict[int, float]] = None,
    ) -> None:
        """
        Args:
            text: An iterable of sentences.
            tokenizer: `AutoTokenizer` instance from `transformers` package.
            max_length: A maximum sequence length.
            tokens_idf: Inverse document frequences (these should be calculated on reference sentences).
        """
        self.text = preprocess_text_fn(text, tokenizer, max_length)
        self.num_sentences = len(text)
        self.tokens_idf = tokens_idf if tokens_idf is not None else self._get_tokens_idf()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids = self.text["input_ids"][idx, :]
        attention_mask = self.text["attention_mask"][idx, :]
        input_ids_idf = torch.tensor([self.tokens_idf[input_idx] for input_idx in input_ids.tolist()])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "input_ids_idf": input_ids_idf}

    def __len__(self) -> int:
        return self.num_sentences

    def _get_tokens_idf(self) -> Dict[int, float]:
        """Calculate token inverse document frequences.

        Return:
            A python dictionary containing inverse document frequences for token ids.
        """
        unique_ids, ids_occurrence = self.text["input_ids"].unique(return_counts=True)

        tokens_idf: Dict[int, float] = defaultdict(lambda: math.log((self.num_sentences + 1) / 1))
        tokens_idf.update(
            {
                idx: math.log((self.num_sentences + 1) / (occurrence + 1))
                for idx, occurrence in zip(unique_ids.tolist(), ids_occurrence.tolist())
            }
        )
        return tokens_idf


def _get_embeddings_and_idf_scale(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    idf: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate sentence embeddings and the inverse-document-frequence scaling factor.
    TODO:
    Args:
        dataloader:
        model:
        device:
        idf:

    Return:
    """
    EMBEDDINGS_LIST: List[torch.Tensor] = []
    IDF_SCALE_LIST: List[torch.Tensor] = []
    for batch in dataloader:
        with torch.no_grad():
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))[0]
        out /= out.norm(dim=-1).unsqueeze(-1)  # normalize embeddings
        # Multiply embeddings with attention_mask (b=batch_size, s=seq_len, d=emb_dim)
        out = torch.einsum("bsd, bs -> bsd", out, batch["attention_mask"])
        EMBEDDINGS_LIST.append(out.cpu())

        # Calculate idf scaling factor if desired. Otherwise take a vector of ones
        idf_scale = (batch["input_ids_idf"] * batch["attention_mask"]).sum(-1) if idf else torch.ones(out.shape[0])
        IDF_SCALE_LIST.append(idf_scale)

    EMBEDDINGS = torch.cat(EMBEDDINGS_LIST)
    IDF_SCALE = torch.cat(IDF_SCALE_LIST)

    return EMBEDDINGS, IDF_SCALE


def _get_precision_recall_f1(
    pred_embeddings: torch.Tensor,
    ref_embeddings: torch.Tensor,
    pred_idf_scale: torch.Tensor,
    ref_idf_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate precision, recall and F1 score over candidate and reference sentences.

    Args:
        pred_embeddings: Embeddings of candidate sentenecs.
        ref_embeddings: Embeddings of reference sentences.
        pred_idf_scale: An IDF scale factor for candidate sentences.
        ref_idf_scale: An IDF scale factor for reference sentences.

    Return:
        Tensors containing precision, recall and F1 score, respectively.
    """
    cos_sim = torch.bmm(pred_embeddings, ref_embeddings.transpose(1, 2))
    precision = cos_sim.max(dim=2).values.mean(-1) / pred_idf_scale
    recall = cos_sim.max(dim=1).values.mean(-1) / ref_idf_scale
    f1_score = 2 * precision * recall / (precision * recall)
    f1_score = f1_score.masked_fill(torch.isnan(f1_score), 0.0)

    return precision, recall, f1_score


def new_bert_score(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
    model_type: Optional[str] = None,
    model_path: Optional[str] = None,
    own_model: Optional[torch.nn.Module] = None,
    idf: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    max_length: int = 512,
    batch_size: int = 64,
    num_threads: int = 4,
) -> Dict[str, List[float]]:
    if not own_model:
        if not _TRANSFORMERS_AVAILABLE:
            raise ValueError("#TODO: Transformers must be installed.")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.eval()
        model.to(device)

    ref_dataset = TextDataset(references, tokenizer, max_length)
    pred_dataset = TextDataset(predictions, tokenizer, max_length, tokens_idf=ref_dataset.tokens_idf)
    ref_loader = DataLoader(ref_dataset, batch_size=batch_size, num_workers=num_threads)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, num_workers=num_threads)

    ref_embeddings, ref_idf_scale = _get_embeddings_and_idf_scale(ref_loader, model, device, idf)
    pred_embeddings, pred_idf_scale = _get_embeddings_and_idf_scale(pred_loader, model, device, idf)

    precision, recall, f1_score = _get_precision_recall_f1(
        pred_embeddings, ref_embeddings, pred_idf_scale, ref_idf_scale
    )

    output_dict = {
        "f1": f1_score.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
    }
    return output_dict


def bert_score(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
    model_type: Optional[str] = None,
    num_layers: int = None,
    verbose: bool = False,
    idf: bool = False,
    device: Optional[str] = None,
    batch_size: int = 64,
    num_threads: int = 4,
    all_layers: bool = False,
    rescale_with_baseline: bool = False,
    baseline_path: Optional[str] = None,
) -> Dict:
    """`BERTScore <https://arxiv.org/abs/1904.09675>`_ leverages the pre-trained contextual embeddings from BERT
    and matches words in candidate and reference sentences by cosine similarity. It has been shown to correlate
    with human judgment on sentence-level and system-level evaluation. Moreover, BERTScore computes precision,
    recall, and F1 measure, which can be useful for evaluating different language generation tasks.

    Args:
        predictions: candidate sentences
        references: reference sentences
        model_type: bert specification
        num_layers: the layer of representation to use.
        verbose: turn on intermediate status update
        idf: use idf weighting, can also be a precomputed idf_dict
        device: on which the contextual embedding model will be allocated on.
        num_threads: number of threads
        batch_size: bert score processing batch size
        lang: language of the sentences
        rescale_with_baseline: rescale bertscore with pre-computed baseline
        baseline_path: customized baseline file

    Returns:
        Dict containing the keys `precision`, `recall`, `f1` and `hashcode` with corresponding values

    Example:
        >>> predictions = ["hello there", "general kenobi"]
        >>> references = ["hello there", "master kenobi"]
        >>> bert_score(predictions=predictions, references=references, lang="en")  # doctest: +SKIP
        {'f1': [0.99..., 0.99...],
         'hashcode': '...',
         'precision': [0.99..., 0.99...],
         'recall': [0.99..., 0.99...]}
    """

    if not _BERTSCORE_AVAILABLE:
        raise ValueError(
            "bert_score metric requires that bert-score package is installed."
            " Either install with `pip install bert-score` or `pip install torchmetrics[text]`"
        )

    if model_type is None:
        model_type = lang2model[lang.lower()]

    if num_layers is None:
        num_layers = model2layers[model_type]

    hashcode = get_hash(
        model=model_type,
        num_layers=num_layers,
        idf=idf,
        rescale_with_baseline=rescale_with_baseline,
        use_custom_baseline=baseline_path is not None,
        use_fast_tokenizer=True,
    )

    cached_bertscorer = BERTScorer(
        model_type=model_type,
        num_layers=num_layers,
        batch_size=batch_size,
        nthreads=num_threads,
        all_layers=all_layers,
        idf=idf,
        device=device,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        baseline_path=baseline_path,
    )

    prec, recall, f1 = cached_bertscorer.score(
        cands=predictions,
        refs=references,
        verbose=verbose,
        batch_size=batch_size,
    )
    output_dict = {
        "precision": prec.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "hashcode": hashcode,
    }
    return output_dict

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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from torchmetrics.functional.text.helper_embedding_metric import _load_tokenizer_and_model
from torchmetrics.functional.text.infolm import (
    _ALLOWED_INFORMATION_MEASURE_LITERAL,
    _get_dataloader,
    _get_special_tokens_map,
    _infolm_compute,
    _infolm_update,
    _InformationMeasure,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TRANSFORMERS_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["InfoLM.plot"]

if not _TRANSFORMERS_AVAILABLE:
    __doctest_skip__ = ["InfoLM", "InfoLM.plot"]


class InfoLM(Metric):
    """Calculate `InfoLM`_.

    InfoLM measures a distance/divergence between predicted and reference sentence discrete distribution using one of
    the following information measures:

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

    The implementation of this metric is fully based HuggingFace ``transformers``' package.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~Sequence`): An iterable of hypothesis corpus
    - ``target`` (:class:`~Sequence`): An iterable of reference corpus

    As output of ``forward`` and ``compute`` the metric returns the following output:

    -  ``infolm`` (:class:`~torch.Tensor`): If `return_sentence_level_score=True` return a tuple with a tensor
       with the corpus-level InfoLM score and a list of sentence-level InfoLM scores, else return a corpus-level
       InfoLM score

    Args:
        model_name_or_path:
            A name or a model path used to load ``transformers`` pretrained model.
            By default the `"bert-base-uncased"` model is used.
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
            A maximum length of input sequences. Sequences longer than ``max_length`` are to be trimmed.
        batch_size:
            A batch size used for model processing.
        num_threads:
            A number of threads to use for a dataloader.
        verbose:
            An indication of whether a progress bar to be displayed during the embeddings calculation.
        return_sentence_level_score:
            An indication whether a sentence-level InfoLM score to be returned.

    Example:
        >>> from torchmetrics.text.infolm import InfoLM
        >>> preds = ['he read the book because he was interested in world history']
        >>> target = ['he was interested in world history because he read the book']
        >>> infolm = InfoLM('google/bert_uncased_L-2_H-128_A-2', idf=False)
        >>> infolm(preds, target)
        tensor(-0.1784)
    """

    is_differentiable = False
    higher_is_better = True
    preds_input_ids: List[Tensor]
    preds_attention_mask: List[Tensor]
    target_input_ids: List[Tensor]
    target_attention_mask: List[Tensor]

    def __init__(
        self,
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
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.temperature = temperature
        self.information_measure = information_measure
        self.idf = idf
        self.alpha = alpha
        self.beta = beta
        self._device = torch.device(device or "cpu")
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.verbose = verbose
        self.return_sentence_level_score = return_sentence_level_score

        self.tokenizer, self.model = _load_tokenizer_and_model(model_name_or_path, device)
        self.information_measure_cls = _InformationMeasure(information_measure, alpha, beta)
        self.max_length = max_length or self.model.config.max_length
        self.special_tokens_map = _get_special_tokens_map(self.tokenizer)

        self.add_state("preds_input_ids", [], dist_reduce_fx="cat")
        self.add_state("preds_attention_mask", [], dist_reduce_fx="cat")
        self.add_state("target_input_ids", [], dist_reduce_fx="cat")
        self.add_state("target_attention_mask", [], dist_reduce_fx="cat")

    def update(self, preds: Union[str, Sequence[str]], target: Union[str, Sequence[str]]) -> None:
        """Update state with predictions and targets."""
        preds_input_ids, preds_attention_mask, target_input_ids, target_attention_mask = _infolm_update(
            preds, target, self.tokenizer, self.max_length
        )
        self.preds_input_ids.append(preds_input_ids)
        self.preds_attention_mask.append(preds_attention_mask)
        self.target_input_ids.append(target_input_ids)
        self.target_attention_mask.append(target_attention_mask)

    def compute(self) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Calculate selected information measure using the pre-trained language model."""
        preds_dataloader = _get_dataloader(
            input_ids=dim_zero_cat(self.preds_input_ids),
            attention_mask=dim_zero_cat(self.preds_attention_mask),
            idf=self.idf,
            batch_size=self.batch_size,
            num_workers=self.num_threads,
        )
        target_dataloader = _get_dataloader(
            input_ids=dim_zero_cat(self.target_input_ids),
            attention_mask=dim_zero_cat(self.target_attention_mask),
            idf=self.idf,
            batch_size=self.batch_size,
            num_workers=self.num_threads,
        )

        info_lm_score = _infolm_compute(
            self.model,
            preds_dataloader,
            target_dataloader,
            self.temperature,
            self.idf,
            self.information_measure_cls,
            self.special_tokens_map,
            self.verbose,
        )

        if self.return_sentence_level_score:
            return info_lm_score.mean(), info_lm_score

        return info_lm_score.mean()

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> from torchmetrics.text.infolm import InfoLM
            >>> metric = InfoLM('google/bert_uncased_L-2_H-128_A-2', idf=False)
            >>> preds = ['he read the book because he was interested in world history']
            >>> target = ['he was interested in world history because he read the book']
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torchmetrics.text.infolm import InfoLM
            >>> metric = InfoLM('google/bert_uncased_L-2_H-128_A-2', idf=False)
            >>> preds = ["this is the prediction", "there is an other sample"]
            >>> target = ["this is the reference", "there is another one"]
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(preds, target))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)

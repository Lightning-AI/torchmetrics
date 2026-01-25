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
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor
from torch.nn import Module

from torchmetrics.functional.text.helper_embedding_metric import _preprocess_text
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE
from torchmetrics.utilities.data import dim_zero_cat

from torchmetrics.functional.text.depth_score import (
    _postprocess_multiple_references_distance,
    _preprocess_multiple_references,
    depth_score,
)

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["DepthScore.plot"]

# Default model recommended in the original implementation.
_DEFAULT_MODEL: str = "bert-base-uncased"

if _SKIP_SLOW_DOCTEST and _TRANSFORMERS_GREATER_EQUAL_4_4:
    from transformers import AutoModel, AutoTokenizer

    def _download_model_for_depth_score() -> None:
        """Download intensive operations."""
        AutoTokenizer.from_pretrained(_DEFAULT_MODEL, resume_download=True)
        AutoModel.from_pretrained(_DEFAULT_MODEL, resume_download=True)

    if not _try_proceed_with_timeout(_download_model_for_depth_score):
        __doctest_skip__ = ["DepthScore", "DepthScore.plot"]
else:
    __doctest_skip__ = ["DepthScore", "DepthScore.plot"]


def _get_input_dict(input_ids: List[Tensor], attention_mask: List[Tensor]) -> dict[str, Tensor]:
    """Create an input dictionary of ``input_ids`` and ``attention_mask`` for DepthScore calculation."""
    return {"input_ids": torch.cat(input_ids), "attention_mask": torch.cat(attention_mask)}


class DepthScore(Metric):
    """`DepthScore Evaluating Text Generation`_ for measuring text similarity.

    DepthScore leverages pre-trained contextual token embeddings (e.g., from BERT-like models) and compares candidate and
    reference sentences by treating their token embeddings as point clouds and computing a depth-based pseudo-metric
    between the two distributions. This distance is designed to capture distributional mismatches between contextual
    representations and can be used for evaluating text generation tasks where *lower* distance indicates a better match.
    This implementation follows the reference DepthScore formulation introduced by Colombo et al. and mirrors the
    TorchMetrics-style API used by embedding-based text metrics.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds``: Predicted sentence(s). Can be one of:

        * A single predicted sentence as a string (``str``)
        * A sequence of predicted sentences (``Sequence[str]``)

    - ``target``: Target/reference sentence(s). Can be one of:

        * A single reference sentence as a string (``str``)
        * A sequence of reference sentences (``Sequence[str]``)
        * A sequence of sequences of reference sentences for multi-reference evaluation (``Sequence[Sequence[str]]``)

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``score`` (:class:`~torch.Tensor`): A 1D tensor of distances of shape `(num_predictions,)`. For multi-reference input,
      the output is reduced per original prediction according to `multi_ref_reduction`.

    Args:
        preds (Union[str, Sequence[str]]): A single predicted sentence or a sequence of predicted sentences.
        target (Union[str, Sequence[str], Sequence[Sequence[str]]]): A single target sentence, a sequence of target
            sentences, or a sequence of sequences of target sentences for multiple references per prediction.
        model_name_or_path: A name or a model path used to load ``transformers`` pretrained model.
        num_layers: A layer of representation to use.
        all_layers:
            An indication of whether the representation from all model's layers should be used.
            If ``all_layers=True``, the argument ``num_layers`` is ignored.
        model: A user's own model. Must be of `torch.nn.Module` instance.
        user_tokenizer:
            A user's own tokenizer used with the own model. This must be an instance with the ``__call__`` method.
            This method must take an iterable of sentences (`List[str]`) and must return a python dictionary
            containing `"input_ids"` and `"attention_mask"` represented by :class:`~torch.Tensor`.
            It is up to the user's model of whether `"input_ids"` is a :class:`~torch.Tensor` of input ids or embedding
            vectors. This tokenizer must prepend an equivalent of ``[CLS]`` token and append an equivalent of ``[SEP]``
            token as ``transformers`` tokenizer does.
        user_forward_fn:
            A user's own forward function used in a combination with ``user_model``. This function must take
            ``user_model`` and a python dictionary of containing ``"input_ids"`` and ``"attention_mask"`` represented
            by :class:`~torch.Tensor` as an input and return the model's output represented by the single
            :class:`~torch.Tensor`.
        verbose: An indication of whether a progress bar to be displayed during the embeddings' calculation.
        device: A device to be used for calculation.
        max_length: A maximum length of input sequences. Sequences longer than ``max_length`` are to be trimmed.
        batch_size: A batch size used for model processing.
        num_threads: A number of threads to use for a dataloader.
        n_alpha: The Monte-Carlo parameter for the approximation of the integral over alpha (number of level-set
            thresholds between ``eps`` and 1.0).
        eps: The lowest level-set bound in [0, 1]. The highest level set is fixed to 1.0 in this implementation.
        p: The power of the ground cost.
        measure: Depth / discrepancy measure to use (e.g. ``"irw"`` or ``"ai_irw"``).
        truncation: An indication of whether the input sequences should be truncated to the ``max_length``.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Example:
        >>> from pprint import pprint
        >>> from torchmetrics.text.depth_score import DepthScore
        >>> preds = ["hello there", "general kenobi"]
        >>> target = ["hello there", "master kenobi"]
        >>> depthscore = DepthScore()
        >>> pprint(depthscore(preds, target))
        tensor([...])

    Example:
        >>> from pprint import pprint
        >>> from torchmetrics.text.depth_score import DepthScore
        >>> preds = ["hello there", "general kenobi"]
        >>> target = [["hello there", "master kenobi"], ["hello there", "master kenobi"]]
        >>> depthscore = DepthScore()
        >>> pprint(depthscore(preds, target))
        tensor([...])

    """

    is_differentiable: bool = False
    higher_is_better: bool = False  # distance
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0  # not truly bounded; used only for plotting convenience

    preds_input_ids: List[Tensor]
    preds_attention_mask: List[Tensor]
    target_input_ids: List[Tensor]
    target_attention_mask: List[Tensor]

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        num_layers: Optional[int] = None,
        all_layers: bool = False,
        model: Optional[Module] = None,
        user_tokenizer: Optional[Any] = None,
        user_forward_fn: Optional[Callable[[Module, dict[str, Tensor]], Tensor]] = None,
        verbose: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 512,
        batch_size: int = 64,
        num_threads: int = 0,
        n_alpha: int = 5,
        eps: float = 0.3,
        p: int = 5,
        measure: str = "irw",
        truncation: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not _TRANSFORMERS_GREATER_EQUAL_4_4 and user_tokenizer is None:
            raise ModuleNotFoundError(
                "`DepthScore` metric with default tokenizers requires `transformers` package be installed."
                " Either install with `pip install transformers>=4.4` or `pip install torchmetrics[text]`."
            )

        self.model_name_or_path = model_name_or_path or _DEFAULT_MODEL
        self.num_layers = num_layers
        self.all_layers = all_layers
        self.model = model
        self.user_forward_fn = user_forward_fn
        self.verbose = verbose
        self.embedding_device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.n_alpha = n_alpha
        self.eps = eps
        self.p = p
        self.measure = measure
        self.truncation = truncation

        self.ref_group_boundaries: Optional[List[Tuple[int, int]]] = None

        if user_tokenizer:
            self.tokenizer = user_tokenizer
            self.user_tokenizer = True
        else:
            from transformers import AutoTokenizer

            if model_name_or_path is None:
                rank_zero_warn(
                    "The argument `model_name_or_path` was not specified while it is required when the default"
                    f" `transformers` model is used. It will use the default recommended model - {_DEFAULT_MODEL!r}."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.user_tokenizer = False

        self.add_state("preds_input_ids", [], dist_reduce_fx="cat")
        self.add_state("preds_attention_mask", [], dist_reduce_fx="cat")
        self.add_state("target_input_ids", [], dist_reduce_fx="cat")
        self.add_state("target_attention_mask", [], dist_reduce_fx="cat")

    def update(
        self, preds: Union[str, Sequence[str]], target: Union[str, Sequence[str], Sequence[Sequence[str]]]
    ) -> None:
        """Store predictions/references for computing DepthScore.

        It is necessary to store sentences in a tokenized form to ensure the DDP mode working.
        """
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(target, str):
            target = [target]
        if not isinstance(preds, list):
            preds = list(preds)
        if not isinstance(target, list):
            target = list(target)

        if len(preds) != len(target):
            raise ValueError(
                "Expected number of predicted and reference sentences to be the same, but got"
                f"{len(preds)} and {len(target)}"
            )

        if isinstance(preds, list) and len(preds) > 0 and isinstance(target, list) and len(target) > 0:
            preds, target, self.ref_group_boundaries = _preprocess_multiple_references(preds, target)

        preds_dict, _ = _preprocess_text(
            preds,
            self.tokenizer,
            self.max_length,
            truncation=self.truncation,
            sort_according_length=False,
            own_tokenizer=self.user_tokenizer,
        )
        target_dict, _ = _preprocess_text(
            cast(List[str], target),
            self.tokenizer,
            self.max_length,
            truncation=self.truncation,
            sort_according_length=False,
            own_tokenizer=self.user_tokenizer,
        )

        self.preds_input_ids.append(preds_dict["input_ids"])
        self.preds_attention_mask.append(preds_dict["attention_mask"])
        self.target_input_ids.append(target_dict["input_ids"])
        self.target_attention_mask.append(target_dict["attention_mask"])

    def compute(self) -> Tensor:
        """Calculate DepthScore."""
        preds = {
            "input_ids": dim_zero_cat(self.preds_input_ids),
            "attention_mask": dim_zero_cat(self.preds_attention_mask),
        }
        target = {
            "input_ids": dim_zero_cat(self.target_input_ids),
            "attention_mask": dim_zero_cat(self.target_attention_mask),
        }

        out = depth_score(
            preds=preds,  # supports dict input (tokenized)
            target=target,
            model_name_or_path=self.model_name_or_path,
            num_layers=self.num_layers,
            all_layers=self.all_layers,
            n_alpha=self.n_alpha,
            eps=self.eps,
            p=self.p,
            measure=self.measure,
            device=self.embedding_device if self.embedding_device is not None else None,
            model=self.model,
            user_tokenizer=self.tokenizer if self.user_tokenizer else None,
            user_forward_fn=self.user_forward_fn,
            max_length=self.max_length,
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            truncation=self.truncation,
            verbose=self.verbose,
        )

        # out expected: {"depth_score": Tensor} aligned with flattened refs if multi-ref used
        if self.ref_group_boundaries is not None:
            out = _postprocess_multiple_references_distance(
                out,
                self.ref_group_boundaries,
                reduction="min",   # distance metric → best match is smallest distance
            )

        return out

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: A matplotlib axis object. If provided will add plot to that axis.

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> from torchmetrics.text.depth_score import DepthScore
            >>> preds = ["hello there", "general kenobi"]
            >>> target = ["hello there", "master kenobi"]
            >>> metric = DepthScore()
            >>> metric.update(preds, target)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torch import tensor
            >>> from torchmetrics.text.depth_score import DepthScore
            >>> preds = ["hello there", "general kenobi"]
            >>> target = ["hello there", "master kenobi"]
            >>> metric = DepthScore()
            >>> values = []
            >>> for _ in range(10):
            ...     val = metric(preds, target)
            ...     val = val.mean()  # convert into a single scalar
            ...     values.append(val)
            >>> fig_, ax_ = metric.plot(values)

        """
        if val is None:  # default average score across sentences
            val = self.compute()  # type: ignore
            val = val.mean()  # type: ignore
        return self._plot(val, ax)
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
from typing import Any, Callable, Optional, Sequence, Tuple, Union, no_type_check

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.classification.dice import _dice_compute
from torchmetrics.functional.classification.stat_scores import _stat_scores_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

if not _MATPLOTLIB_AVAILABLE:
    __doctest_skip__ = ["Dice.plot"]


class Dice(Metric):
    r"""Compute `Dice`_.

    .. math:: \text{Dice} = \frac{\text{2 * TP}}{\text{2 * TP} + \text{FP} + \text{FN}}

    Where :math:`\text{TP}` and :math:`\text{FP}` represent the number of true positives and
    false positives respecitively.

    It is recommend set `ignore_index` to index of background class.

    The reduction method (how the precision scores are aggregated) is controlled by the
    ``average`` parameter, and additionally by the ``mdmc_average`` parameter in the
    multi-dimensional multi-class case.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model (probabilities, logits or labels)
    - ``target`` (:class:`~torch.Tensor`): Ground truth values

    As output to ``forward`` and ``compute`` the metric returns the following output:

    - ``dice`` (:class:`~torch.Tensor`): A tensor containing the dice score.

        - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
        - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number of classes

    Args:
        num_classes:
            Number of classes. Necessary for ``'macro'``, and ``None`` average methods.
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        zero_division:
            The value to use for the score if denominator equals zero.
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note::
               What is considered a sample in the multi-dimensional multi-class case
               depends on the value of ``mdmc_average``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              are flattened into a new ``N_X`` sample axis, i.e.
              the inputs are treated as if they were ``(N_X, C)``.
              From here on the ``average`` parameter applies as usual.

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.

        top_k:
            Number of the highest probability or logit score predictions considered finding the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.
            Should be left at default (``None``) for all other types of inputs.

        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be.

        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    Raises:
        ValueError:
            If ``average`` is none of ``"micro"``, ``"macro"``, ``"samples"``, ``"none"``, ``None``.
        ValueError:
            If ``mdmc_average`` is not one of ``None``, ``"samplewise"``, ``"global"``.
        ValueError:
            If ``average`` is set but ``num_classes`` is not provided.
        ValueError:
            If ``num_classes`` is set and ``ignore_index`` is not in the range ``[0, num_classes)``.

    Example:
        >>> from torch import tensor
        >>> from torchmetrics.classification import Dice
        >>> preds  = tensor([2, 0, 2, 1])
        >>> target = tensor([1, 1, 2, 0])
        >>> dice = Dice(average='micro')
        >>> dice(preds, target)
        tensor(0.2500)
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0
    plot_legend_name: str = "Class"

    @no_type_check
    def __init__(
        self,
        zero_division: int = 0,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: Optional[Literal["micro", "macro", "none"]] = "micro",
        mdmc_average: Optional[str] = "global",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        allowed_average = ("micro", "macro", "samples", "none", None)
        if average not in allowed_average:
            raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

        _reduce_options = (AverageMethod.WEIGHTED, AverageMethod.NONE, None)
        if "reduce" not in kwargs:
            kwargs["reduce"] = AverageMethod.MACRO if average in _reduce_options else average
        if "mdmc_reduce" not in kwargs:
            kwargs["mdmc_reduce"] = mdmc_average

        self.reduce = average
        self.mdmc_reduce = mdmc_average
        self.num_classes = num_classes
        self.threshold = threshold
        self.multiclass = multiclass
        self.ignore_index = ignore_index
        self.top_k = top_k

        if average not in ["micro", "macro", "samples"]:
            raise ValueError(f"The `reduce` {average} is not valid.")

        if mdmc_average not in [None, "samplewise", "global"]:
            raise ValueError(f"The `mdmc_reduce` {mdmc_average} is not valid.")

        if average == "macro" and (not num_classes or num_classes < 1):
            raise ValueError("When you set `average` as 'macro', you have to provide the number of classes.")

        if num_classes and ignore_index is not None and (not ignore_index < num_classes or num_classes == 1):
            raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

        default: Callable = list
        reduce_fn: Optional[str] = "cat"
        if mdmc_average != "samplewise" and average != "samples":
            if average == "micro":
                zeros_shape = []
            elif average == "macro":
                zeros_shape = [num_classes]
            else:
                raise ValueError(f'Wrong reduce="{average}"')
            default = lambda: torch.zeros(zeros_shape, dtype=torch.long)
            reduce_fn = "sum"

        for s in ("tp", "fp", "tn", "fn"):
            self.add_state(s, default=default(), dist_reduce_fx=reduce_fn)

        self.average = average
        self.zero_division = zero_division

    @no_type_check
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        tp, fp, tn, fn = _stat_scores_update(
            preds,
            target,
            reduce=self.reduce,
            mdmc_reduce=self.mdmc_reduce,
            threshold=self.threshold,
            num_classes=self.num_classes,
            top_k=self.top_k,
            multiclass=self.multiclass,
            ignore_index=self.ignore_index,
        )

        # Update states
        if self.reduce != AverageMethod.SAMPLES and self.mdmc_reduce != MDMCAverageMethod.SAMPLEWISE:
            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn
        else:
            self.tp.append(tp)
            self.fp.append(fp)
            self.tn.append(tn)
            self.fn.append(fn)

    @no_type_check
    def _get_final_stats(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Perform concatenation on the stat scores if neccesary, before passing them to a compute function."""
        tp = torch.cat(self.tp) if isinstance(self.tp, list) else self.tp
        fp = torch.cat(self.fp) if isinstance(self.fp, list) else self.fp
        tn = torch.cat(self.tn) if isinstance(self.tn, list) else self.tn
        fn = torch.cat(self.fn) if isinstance(self.fn, list) else self.fn
        return tp, fp, tn, fn

    @no_type_check
    def compute(self) -> Tensor:
        """Compute metric."""
        tp, fp, _, fn = self._get_final_stats()
        return _dice_compute(tp, fp, fn, self.average, self.mdmc_reduce, self.zero_division)

    def plot(
        self, val: Optional[Union[Tensor, Sequence[Tensor]]] = None, ax: Optional[_AX_TYPE] = None
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure object and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> from torch import randint
            >>> from torchmetrics.classification import Dice
            >>> metric = Dice()
            >>> metric.update(randint(2,(10,)), randint(2,(10,)))
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> from torch import randint
            >>> from torchmetrics.classification import Dice
            >>> metric = Dice()
            >>> values = [ ]
            >>> for _ in range(10):
            ...     values.append(metric(randint(2,(10,)), randint(2,(10,))))
            >>> fig_, ax_ = metric.plot(values)
        """
        return self._plot(val, ax)

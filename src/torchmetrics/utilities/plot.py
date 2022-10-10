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
from itertools import product
from math import ceil, floor, sqrt
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torchmetrics.utilities.compute import _auc_compute_without_check as auc_calc
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE

if _MATPLOTLIB_AVAILABLE:
    import matplotlib
    import matplotlib.pyplot as plt
else:
    plt = None


def _get_col_row_split(n: int) -> Tuple[int, int]:
    """Split n curves into rows x cols figures."""
    nsq = sqrt(n)
    if nsq * nsq == n:
        return nsq, nsq
    elif floor(nsq) * ceil(nsq) > n:
        return floor(nsq), ceil(nsq)
    else:
        return ceil(nsq), ceil(nsq)


def trim_axs(axs: Union[matplotlib.axes.Axes, np.ndarray], N: int) -> np.ndarray:
    """Reduce *axs* to *N* Axes.

    All further Axes are removed from the figure.
    """
    if isinstance(axs, matplotlib.axes.Axes):
        return axs
    else:
        axs = axs.flat
        for ax in axs[N:]:
            ax.remove()
        return axs[:N]


def _error_on_missing_matplotlib() -> None:
    """Raise error if matplotlib is not installed."""
    if not _MATPLOTLIB_AVAILABLE:
        raise ValueError(
            "Plot function expects `matplotlib` to be installed. Please install with `pip install matplotlib`"
        )


def plot_confusion_matrix(
    confmat: Tensor,
    add_text: bool = True,
    labels: Optional[List[str]] = None,
    cmap: Optional[matplotlib.colors.Colormap] = None,
) -> Tuple[plt.Figure, Union[matplotlib.axes.Axes, np.ndarray]]:
    """Inspired by: https://github.com/scikit-learn/scikit-
    learn/blob/main/sklearn/metrics/_plot/confusion_matrix.py."""

    _error_on_missing_matplotlib()

    if confmat.ndim == 3:  # multilabel
        n, n_classes = confmat.shape[0], 2
        rows, cols = _get_col_row_split(n)
    else:
        n, n_classes, rows, cols = 1, confmat.shape[0], 1, 1

    if labels is not None and confmat.ndim != 3 and len(labels) != n_classes:
        raise ValueError(
            "Expected number of elements in arg `labels` to match number of labels in confmat but "
            f"got {len(labels)} and {n_classes}"
        )
    labels = labels if labels is not None else list(range(n_classes))

    fig, axs = plt.subplots(nrows=rows, ncols=cols)
    axs = trim_axs(axs, n)
    for i in range(n):
        if rows != 1 and cols != 1:
            ax = axs[i]
            ax.set_title(f"Label {i}", fontsize=15)
        else:
            ax = axs
        ax.imshow(confmat[i].cpu().detach() if confmat.ndim == 3 else confmat.cpu().detach(), cmap=cmap)
        ax.set_xlabel("True class", fontsize=15)
        ax.set_ylabel("Predicted class", fontsize=15)
        ax.set_xticks(list(range(n_classes)))
        ax.set_yticks(list(range(n_classes)))
        ax.set_xticklabels(labels, rotation=45, fontsize=10)
        ax.set_yticklabels(labels, rotation=25, fontsize=10)

        if add_text:
            for ii, jj in product(range(n_classes), range(n_classes)):
                val = confmat[i, ii, jj] if confmat.ndim == 3 else confmat[ii, jj]
                ax.text(jj, ii, str(val.item()), ha="center", va="center", fontsize=15)

    return fig, axs


def _plot_curve(
    input: Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]],
    auc: bool = False,
    single_plot: bool = False,
    xy_labels: Tuple[str, str] = ("X axis", "Y axis"),
) -> plt.Figure:
    _error_on_missing_matplotlib()
    val1, val2, thresholds = input

    if auc:
        if isinstance(val1, Tensor) and val1.ndim == 1:
            val = auc_calc(val2, val1, 1.0)
        elif isinstance(val1, Tensor):
            val = auc_calc(val2, val1, 1.0, axis=1)
        else:
            val = [auc_calc(x, y, 1.0) for x, y in zip(val2, val1)]
            val = torch.stack(val)

    if isinstance(val1, Tensor) and val1.ndim == 1:  # binary case
        fig, axs = plt.subplots(1, 1)
        label = f"AUC: {val:0.2f}" if auc else ""
        axs.plot(val1.cpu().detach(), val2.cpu().detach(), label=label)
        axs.set_xlabel("False positive rate", fontsize=15)
        axs.set_ylabel("True positive rate", fontsize=15)
        if auc:
            axs.legend()
    else:
        n = len(val1)
        if not single_plot:
            rows, cols = _get_col_row_split(n)
            fig, axs = plt.subplots(rows, cols)
            axs = trim_axs(axs, n)
        else:
            fig, axs = plt.subplots(1, 1)
        for i in range(n):
            ax = axs if single_plot else axs[i]
            label = f"Class/Label {i}" if single_plot else ""
            label += f" AUC: {val[i]:0.2f}" if auc else ""
            ax.plot(val1[i], val2[i], "-", label=label)
            ax.set_xlabel("False positive rate", fontsize=15)
            ax.set_ylabel("True positive rate", fontsize=15)
            if not single_plot:
                ax.set_title(f"Class/Label {i}")
            if label != "":
                ax.legend()

    return fig, axs


def plot_roc(
    roc: Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]],
    auc: bool = False,
    single_plot: bool = False,
) -> plt.Figure:
    return _plot_curve(roc, auc=auc, single_plot=single_plot, xy_labels=("False positive rate", "True positive rate"))


def plot_prc(
    prc: Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]],
    auc: bool = False,
    single_plot: bool = False,
) -> plt.Figure:
    # change order
    return _plot_curve(
        [prc[1].flip(0), prc[0].flip(0), prc[2]], auc=auc, single_plot=single_plot, xy_labels=("Recall", "Precision")
    )

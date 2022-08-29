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
from math import ceil, floor, sqrt
from typing import List, Tuple, Union

from torch import Tensor

from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE

if _MATPLOTLIB_AVAILABLE:
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


def _error_on_missing_matplotlib():
    if not _MATPLOTLIB_AVAILABLE:
        raise ValueError('Plot function expects `matplotlib` to be installed. Please install.')


def plot_confusion_matrix(confmat: Tensor) -> plt.Figure:
    _error_on_missing_matplotlib()
    if confmat.ndims == 3:  # multilabel
        n = confmat.shape[0]
        rows, cols = _get_col_row_split(n)
    else:
        n, rows, cols = 1, 1, 1

    fig, axs = plt.subplots(nrows=rows, ncols=cols)
    for i in range(n):
        axs[i % rows, i % cols].imshow(confmat[i].cpu().detach() if confmat.ndims == 3 else confmat.cpu().detach())
        axs[i % rows, i % cols].xlabel('True class')
        axs[i % rows, i % cols].ylabel('Predicted class')


def plot_roc(
    roc: Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]],
    auc: bool = False,
    single_plot: bool = False,
) -> plt.Figure:
    _error_on_missing_matplotlib()
    tpr, fpr, thresholds = roc
    if isinstance(tpr, Tensor) and tpr.ndim == 1:  # binary
        plt.figure()
        plt.plot(tpr.cpu().detach(), fpr.cpu().detach())
    else:
        if not single_plot:
            rows, cols = _get_col_row_split()
            fig, ax = plt.subplots(rows, cols)


def plot_pr_curve(
    pr: Union[Tuple[Tensor, Tensor, Tensor], Tuple[List[Tensor], List[Tensor], List[Tensor]]],
    auc: bool = False,
    single_plot: bool = True,
) -> plt.Figure:
    _error_on_missing_matplotlib()
    precision, recall, thresholds = pr

    if isinstance()
    if not single_plot:
        rows, cols = _get_col_row_split()
        fig, ax = plt.subplots(rows, cols)

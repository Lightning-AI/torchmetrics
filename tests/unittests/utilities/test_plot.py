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
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from torchmetrics.functional.classification.confusion_matrix import (
    binary_confusion_matrix,
    multiclass_confusion_matrix,
    multilabel_confusion_matrix,
)
from torchmetrics.functional.classification.precision_recall_curve import (
    binary_precision_recall_curve,
    multiclass_precision_recall_curve,
    multilabel_precision_recall_curve,
)
from torchmetrics.functional.classification.roc import binary_roc, multiclass_roc, multilabel_roc
from torchmetrics.utilities.plot import plot_confusion_matrix, plot_prc, plot_roc


@pytest.mark.parametrize(
    "metric, preds, target",
    [
        pytest.param(
            binary_confusion_matrix,
            torch.rand(
                100,
            ),
            torch.randint(2, (100,)),
            id="binary",
        ),
        pytest.param(
            partial(multiclass_confusion_matrix, num_classes=3),
            torch.randint(3, (100,)),
            torch.randint(3, (100,)),
            id="multiclass",
        ),
        pytest.param(
            partial(multilabel_confusion_matrix, num_labels=3),
            torch.randint(2, (100, 3)),
            torch.randint(2, (100, 3)),
            id="multilabel",
        ),
    ],
)
def test_confusion_matrix_plotter(metric, preds, target):
    confmat = metric(preds, target)
    fig, axs = plot_confusion_matrix(confmat)
    assert isinstance(fig, plt.Figure)
    cond1 = isinstance(axs, matplotlib.axes.Axes)
    cond2 = isinstance(axs, np.ndarray) and all(isinstance(a, matplotlib.axes.Axes) for a in axs)
    assert cond1 or cond2


@pytest.mark.parametrize(
    "metric, preds, target, labels",
    [
        pytest.param(
            binary_confusion_matrix,
            torch.rand(
                100,
            ),
            torch.randint(2, (100,)),
            ["cat", "dog"],
            id="binary",
        ),
        pytest.param(
            partial(multiclass_confusion_matrix, num_classes=3),
            torch.randint(3, (100,)),
            torch.randint(3, (100,)),
            ["cat", "dog", "bird"],
            id="multiclass",
        ),
    ],
)
def test_confusion_matrix_plotter_with_labels(metric, preds, target, labels):
    confmat = metric(preds, target)
    fig, axs = plot_confusion_matrix(confmat, labels=labels)
    assert isinstance(fig, plt.Figure)
    cond1 = isinstance(axs, matplotlib.axes.Axes)
    cond2 = isinstance(axs, np.ndarray) and all(isinstance(a, matplotlib.axes.Axes) for a in axs)
    assert cond1 or cond2


@pytest.mark.parametrize(
    "metric, preds, target",
    [
        pytest.param(
            binary_roc,
            torch.rand(
                100,
            ),
            torch.randint(2, (100,)),
            id="binary roc",
        ),
        pytest.param(
            partial(multiclass_roc, num_classes=3),
            torch.randn(100, 3).softmax(-1),
            torch.randint(3, (100,)),
            id="multiclass roc",
        ),
        pytest.param(
            partial(multilabel_roc, num_labels=3),
            torch.rand(100, 3),
            torch.randint(2, (100, 3)),
            id="multilabel roc",
        ),
    ],
)
@pytest.mark.parametrize("auc", [True, False])
@pytest.mark.parametrize("single_plot", [True, False])
def test_roc_plotter(metric, preds, target, auc, single_plot):
    output = metric(preds, target)
    fig, axs = plot_roc(output, auc=auc, single_plot=single_plot)
    assert isinstance(fig, plt.Figure)
    cond1 = isinstance(axs, matplotlib.axes.Axes)
    cond2 = isinstance(axs, np.ndarray) and all(isinstance(a, matplotlib.axes.Axes) for a in axs)
    assert cond1 or cond2


@pytest.mark.parametrize(
    "metric, preds, target",
    [
        pytest.param(
            binary_precision_recall_curve,
            torch.rand(
                100,
            ),
            torch.randint(2, (100,)),
            id="binary prc",
        ),
        pytest.param(
            partial(multiclass_precision_recall_curve, num_classes=3),
            torch.randn(100, 3).softmax(-1),
            torch.randint(3, (100,)),
            id="multiclass prc",
        ),
        pytest.param(
            partial(multilabel_precision_recall_curve, num_labels=3),
            torch.rand(100, 3),
            torch.randint(2, (100, 3)),
            id="multilabel prc",
        ),
    ],
)
@pytest.mark.parametrize("auc", [True, False])
@pytest.mark.parametrize("single_plot", [True, False])
def test_prc_plotter(metric, preds, target, auc, single_plot):
    output = metric(preds, target)
    fig, axs = plot_prc(output, auc=auc, single_plot=single_plot)
    assert isinstance(fig, plt.Figure)
    cond1 = isinstance(axs, matplotlib.axes.Axes)
    cond2 = isinstance(axs, np.ndarray) and all(isinstance(a, matplotlib.axes.Axes) for a in axs)
    assert cond1 or cond2

    import pdb

    pdb.set_trace()

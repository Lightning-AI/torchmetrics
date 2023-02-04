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

from torchmetrics.functional import scale_invariant_signal_noise_ratio
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional.audio.pit import permutation_invariant_training
from torchmetrics.functional.classification.accuracy import binary_accuracy, multiclass_accuracy
from torchmetrics.functional.classification.confusion_matrix import (
    binary_confusion_matrix,
    multiclass_confusion_matrix,
    multilabel_confusion_matrix,
)
from torchmetrics.utilities.plot import plot_confusion_matrix, plot_single_or_multi_val


@pytest.mark.parametrize(
    "metric, preds, target",
    [
        pytest.param(
            binary_accuracy,
            lambda: torch.rand(100),
            lambda: torch.randint(2, (100,)),
            id="binary",
        ),
        pytest.param(
            partial(multiclass_accuracy, num_classes=3),
            lambda: torch.randint(3, (100,)),
            lambda: torch.randint(3, (100,)),
            id="multiclass",
        ),
        pytest.param(
            partial(multiclass_accuracy, num_classes=3, average=None),
            lambda: torch.randint(3, (100,)),
            lambda: torch.randint(3, (100,)),
            id="multiclass and average=None",
        ),
        pytest.param(
            partial(perceptual_evaluation_speech_quality, fs=8000, mode="nb"),
            lambda: torch.randn(8000),
            lambda: torch.randn(8000),
            id="perceptual_evaluation_speech_quality",
        )
    ],
)
@pytest.mark.parametrize("num_vals", [1, 5, 10])
def test_single_multi_val_plotter(metric, preds, target, num_vals):
    vals = []
    for i in range(num_vals):
        vals.append(metric(preds(), target()))
    vals = vals[0] if i == 1 else vals
    fig, ax = plot_single_or_multi_val(vals)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)

@pytest.mark.parametrize(
    "metric, preds, target",
    [
        pytest.param(
            partial(permutation_invariant_training, metric_func=scale_invariant_signal_noise_ratio, eval_func="max"),
            lambda: torch.randn(3, 2, 5),
            lambda: torch.randn(3, 2, 5),
            id="permutation_invariant_training",
        )
    ],
)
@pytest.mark.parametrize("num_vals", [1, 5, 10])
def test_single_multi_val_plotter_pit(metric, preds, target, num_vals):
    vals = []
    for i in range(num_vals):
        vals.append(metric(preds(), target())[0])
    vals = vals[0] if i == 1 else vals
    fig, ax = plot_single_or_multi_val(vals)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)


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

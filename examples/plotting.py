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
import argparse

import matplotlib.pyplot as plt
import torch


def accuracy_example():
    from torchmetrics.classification import MulticlassAccuracy

    p = lambda: torch.randn(20, 5)
    t = lambda: torch.randint(5, (20,))

    # plot single value
    metric = MulticlassAccuracy(num_classes=5)
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot a value per class
    metric = MulticlassAccuracy(num_classes=5, average=None)
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot two values as a series
    metric = MulticlassAccuracy(num_classes=5)
    val1 = metric(p(), t())
    val2 = metric(p(), t())
    fig, ax = metric.plot([val1, val2])

    # plot a series of values per class
    metric = MulticlassAccuracy(num_classes=5, average=None)
    val1 = metric(p(), t())
    val2 = metric(p(), t())
    fig, ax = metric.plot([val1, val2])
    return fig, ax


def mean_squared_error_example():
    from torchmetrics.regression import MeanSquaredError

    p = lambda: torch.randn(20)
    t = lambda: torch.randn(20)

    # single val
    metric = MeanSquaredError()
    metric.update(p(), t())
    fig, ax = metric.plot()

    # multiple values
    metric = MeanSquaredError()
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)
    return fig, ax


def confusion_matrix_example():
    from torchmetrics.classification import MulticlassConfusionMatrix

    p = lambda: torch.randn(20, 5)
    t = lambda: torch.randint(5, (20,))

    # plot single value
    metric = MulticlassConfusionMatrix(num_classes=5)
    metric.update(p(), t())
    fig, ax = metric.plot()
    return fig, ax


def spectral_distortion_index_example():
    from torchmetrics.image.d_lambda import SpectralDistortionIndex

    p = lambda: torch.rand([16, 3, 16, 16])
    t = lambda: torch.rand([16, 3, 16, 16])

    # plot single value
    metric = SpectralDistortionIndex()
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = SpectralDistortionIndex()
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)

    return fig, ax


def error_relative_global_dimensionless_synthesis():
    from torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis

    p = lambda: torch.rand([16, 1, 16, 16], generator=torch.manual_seed(42))
    t = lambda: torch.rand([16, 1, 16, 16], generator=torch.manual_seed(42))

    # plot single value
    metric = ErrorRelativeGlobalDimensionlessSynthesis()
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = ErrorRelativeGlobalDimensionlessSynthesis()
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)

    return fig, ax


if __name__ == "__main__":
    metrics_func = {
        "accuracy": accuracy_example,
        "mean_squared_error": mean_squared_error_example,
        "confusion_matrix": confusion_matrix_example,
        "spectral_distortion_index": spectral_distortion_index_example,
        "error_relative_global_dimensionless_synthesis": error_relative_global_dimensionless_synthesis,
    }

    parser = argparse.ArgumentParser(description="Example script for plotting metrics.")
    parser.add_argument("metric", type=str, nargs="?", choices=list(metrics_func.keys()), default="accuracy")
    args = parser.parse_args()

    fig, ax = metrics_func[args.metric]()

    plt.show()

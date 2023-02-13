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
import argparse

import matplotlib.pyplot as plt
import torch


def accuracy_example():
    """Plot Accuracy example."""
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
    """Plot mean squared error example."""
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
    """Plot confusion matrix example."""
    from torchmetrics.classification import MulticlassConfusionMatrix

    p = lambda: torch.randn(20, 5)
    t = lambda: torch.randint(5, (20,))

    # plot single value
    metric = MulticlassConfusionMatrix(num_classes=5)
    metric.update(p(), t())
    fig, ax = metric.plot()
    return fig, ax


def spectral_distortion_index_example():
    """Plot spectral distortion index example example."""
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
    """Plot error relative global dimensionless synthesis example."""
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


def peak_signal_noise_ratio():
    """Plot peak signal noise ratio example."""
    from torchmetrics.image.psnr import PeakSignalNoiseRatio

    p = lambda: torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    t = lambda: torch.tensor([[3.0, 2.0], [1.0, 0.0]])

    # plot single value
    metric = PeakSignalNoiseRatio()
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = PeakSignalNoiseRatio()
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)

    return fig, ax


def spectral_angle_mapper():
    """Plot spectral angle mapper example."""
    from torchmetrics.image.sam import SpectralAngleMapper

    p = lambda: torch.rand([16, 3, 16, 16], generator=torch.manual_seed(42))
    t = lambda: torch.rand([16, 3, 16, 16], generator=torch.manual_seed(123))

    # plot single value
    metric = SpectralAngleMapper()
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = SpectralAngleMapper()
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)

    return fig, ax


def structural_similarity_index_measure():
    """Plot structural similarity index measure example."""
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

    p = lambda: torch.rand([3, 3, 256, 256], generator=torch.manual_seed(42))
    t = lambda: p() * 0.75

    # plot single value
    metric = StructuralSimilarityIndexMeasure()
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = StructuralSimilarityIndexMeasure()
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)

    return fig, ax


def multiscale_structural_similarity_index_measure():
    """Plot multiscale structural similarity index measure example."""
    from torchmetrics.image.ssim import MultiScaleStructuralSimilarityIndexMeasure

    p = lambda: torch.rand([3, 3, 256, 256], generator=torch.manual_seed(42))
    t = lambda: p() * 0.75

    # plot single value
    metric = MultiScaleStructuralSimilarityIndexMeasure()
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = MultiScaleStructuralSimilarityIndexMeasure()
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)

    return fig, ax


def universal_image_quality_index():
    """Plot universal image quality index example."""
    from torchmetrics.image.uqi import UniversalImageQualityIndex

    p = lambda: torch.rand([16, 1, 16, 16])
    t = lambda: p() * 0.75

    # plot single value
    metric = UniversalImageQualityIndex()
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = UniversalImageQualityIndex()
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
        "peak_signal_noise_ratio": peak_signal_noise_ratio,
        "spectral_angle_mapper": spectral_angle_mapper,
        "structural_similarity_index_measure": structural_similarity_index_measure,
        "multiscale_structural_similarity_index_measure": multiscale_structural_similarity_index_measure,
        "universal_image_quality_index": universal_image_quality_index,
    }

    parser = argparse.ArgumentParser(description="Example script for plotting metrics.")
    parser.add_argument("metric", type=str, nargs="?", choices=list(metrics_func.keys()), default="accuracy")
    args = parser.parse_args()

    fig, ax = metrics_func[args.metric]()

    plt.show()

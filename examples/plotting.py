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


def pesq_example():
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

    p = lambda: torch.randn(8000)
    t = lambda: torch.randn(8000)

    # plot single value
    metric = PerceptualEvaluationSpeechQuality(8000, "nb")
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = PerceptualEvaluationSpeechQuality(16000, "wb")
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)

    return fig, ax


def pit_example():
    from torchmetrics.audio.pit import PermutationInvariantTraining
    from torchmetrics.functional import scale_invariant_signal_noise_ratio

    p = lambda: torch.randn(3, 2, 5)
    t = lambda: torch.randn(3, 2, 5)

    # plot single value
    metric = PermutationInvariantTraining(scale_invariant_signal_noise_ratio, "max")
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = PermutationInvariantTraining(scale_invariant_signal_noise_ratio, "max")
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)

    return fig, ax


def sdr_example():
    from torchmetrics.audio.sdr import SignalDistortionRatio

    p = lambda: torch.randn(8000)
    t = lambda: torch.randn(8000)

    # plot single value
    metric = SignalDistortionRatio()
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = SignalDistortionRatio()
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)

    return fig, ax


def si_sdr_example():
    from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio

    p = lambda: torch.randn(5)
    t = lambda: torch.randn(5)

    # plot single value
    metric = ScaleInvariantSignalDistortionRatio()
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = ScaleInvariantSignalDistortionRatio()
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)

    return fig, ax


def snr_example():
    from torchmetrics.audio.snr import SignalNoiseRatio

    p = lambda: torch.randn(4)
    t = lambda: torch.randn(4)

    # plot single value
    metric = SignalNoiseRatio()
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = SignalNoiseRatio()
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)

    return fig, ax


def si_snr_example():
    from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio

    p = lambda: torch.randn(4)
    t = lambda: torch.randn(4)

    # plot single value
    metric = ScaleInvariantSignalNoiseRatio()
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = ScaleInvariantSignalNoiseRatio()
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)

    return fig, ax


def stoi_example():
    from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

    p = lambda: torch.randn(8000)
    t = lambda: torch.randn(8000)

    # plot single value
    metric = ShortTimeObjectiveIntelligibility(8000, False)
    metric.update(p(), t())
    fig, ax = metric.plot()

    # plot multiple values
    metric = ShortTimeObjectiveIntelligibility(8000, False)
    vals = [metric(p(), t()) for _ in range(10)]
    fig, ax = metric.plot(vals)

    return fig, ax


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


if __name__ == "__main__":

    metrics_func = {
        "accuracy": accuracy_example,
        "pesq": pesq_example,
        "pit": pit_example,
        "sdr": sdr_example,
        "si-sdr": si_sdr_example,
        "snr": snr_example,
        "si-snr": si_snr_example,
        "stoi": stoi_example,
        "mean_squared_error": mean_squared_error_example,
        "confusion_matrix": confusion_matrix_example,
    }

    parser = argparse.ArgumentParser(description="Example script for plotting metrics.")
    parser.add_argument("metric", type=str, nargs="?", choices=list(metrics_func.keys()), default="accuracy")
    args = parser.parse_args()

    fig, ax = metrics_func[args.metric]()

    plt.show()

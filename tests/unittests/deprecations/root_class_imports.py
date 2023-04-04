"""Test that domain metric with import from root raises deprecation warning."""
from functools import partial

import pytest

from torchmetrics import (
    PermutationInvariantTraining,
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
    SignalDistortionRatio,
    SignalNoiseRatio,
)
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


@pytest.mark.parametrize(
    "metric_cls",
    [
        pytest.param(
            partial(PermutationInvariantTraining, scale_invariant_signal_noise_ratio), id="PermutationInvariantTraining"
        ),
        ScaleInvariantSignalDistortionRatio,
        ScaleInvariantSignalNoiseRatio,
        SignalDistortionRatio,
        SignalNoiseRatio,
    ],
)
def test_import_from_root_package(metric_cls):
    """Test that domain metric with import from root raises deprecation warning."""
    with pytest.raises(FutureWarning):
        metric_cls()

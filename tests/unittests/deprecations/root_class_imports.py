"""Test that domain metric with import from root raises deprecation warning."""
from functools import partial

import pytest

from torchmetrics import (
    ErrorRelativeGlobalDimensionlessSynthesis,
    ModifiedPanopticQuality,
    MultiScaleStructuralSimilarityIndexMeasure,
    PanopticQuality,
    PeakSignalNoiseRatio,
    PermutationInvariantTraining,
    RelativeAverageSpectralError,
    RootMeanSquaredErrorUsingSlidingWindow,
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
    SignalDistortionRatio,
    SignalNoiseRatio,
    SpectralAngleMapper,
    SpectralDistortionIndex,
    StructuralSimilarityIndexMeasure,
    TotalVariation,
    UniversalImageQualityIndex,
)
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio


@pytest.mark.parametrize(
    "metric_cls",
    [
        # Audio
        pytest.param(
            partial(PermutationInvariantTraining, scale_invariant_signal_noise_ratio), id="PermutationInvariantTraining"
        ),
        ScaleInvariantSignalDistortionRatio,
        ScaleInvariantSignalNoiseRatio,
        SignalDistortionRatio,
        SignalNoiseRatio,
        # Detection
        ModifiedPanopticQuality,
        PanopticQuality,
        # Image
        ErrorRelativeGlobalDimensionlessSynthesis,
        MultiScaleStructuralSimilarityIndexMeasure,
        PeakSignalNoiseRatio,
        RelativeAverageSpectralError,
        RootMeanSquaredErrorUsingSlidingWindow,
        SpectralAngleMapper,
        SpectralDistortionIndex,
        StructuralSimilarityIndexMeasure,
        TotalVariation,
        UniversalImageQualityIndex,
    ],
)
def test_import_from_root_package(metric_cls):
    """Test that domain metric with import from root raises deprecation warning."""
    with pytest.warns(FutureWarning, match=r".+ was deprecated and will be removed in 2.0.+"):
        metric_cls()

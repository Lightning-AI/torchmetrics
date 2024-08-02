"""Test that domain metric with import from root raises deprecation warning."""

from functools import partial

import pytest
from torchmetrics import (
    BLEUScore,
    CharErrorRate,
    CHRFScore,
    ErrorRelativeGlobalDimensionlessSynthesis,
    ExtendedEditDistance,
    MatchErrorRate,
    ModifiedPanopticQuality,
    MultiScaleStructuralSimilarityIndexMeasure,
    PanopticQuality,
    PeakSignalNoiseRatio,
    PermutationInvariantTraining,
    Perplexity,
    RelativeAverageSpectralError,
    RetrievalFallOut,
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalPrecisionRecallCurve,
    RetrievalRecall,
    RetrievalRecallAtFixedPrecision,
    RetrievalRPrecision,
    RootMeanSquaredErrorUsingSlidingWindow,
    SacreBLEUScore,
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
    SignalDistortionRatio,
    SignalNoiseRatio,
    SpectralAngleMapper,
    SpectralDistortionIndex,
    SQuAD,
    StructuralSimilarityIndexMeasure,
    TotalVariation,
    TranslationEditRate,
    UniversalImageQualityIndex,
    WordErrorRate,
    WordInfoLost,
    WordInfoPreserved,
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
        # Info Retrieval
        RetrievalFallOut,
        RetrievalHitRate,
        RetrievalMAP,
        RetrievalMRR,
        RetrievalNormalizedDCG,
        RetrievalPrecision,
        RetrievalPrecisionRecallCurve,
        RetrievalRecall,
        RetrievalRecallAtFixedPrecision,
        RetrievalRPrecision,
        # Text
        BLEUScore,
        CharErrorRate,
        CHRFScore,
        ExtendedEditDistance,
        MatchErrorRate,
        Perplexity,
        SacreBLEUScore,
        SQuAD,
        TranslationEditRate,
        WordErrorRate,
        WordInfoLost,
        WordInfoPreserved,
    ],
)
def test_import_from_root_package(metric_cls):
    """Test that domain metric with import from root raises deprecation warning."""
    with pytest.warns(FutureWarning, match=r".+ was deprecated and will be removed in 2.0.+"):
        metric_cls()

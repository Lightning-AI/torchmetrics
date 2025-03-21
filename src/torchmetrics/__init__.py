r"""Root package info."""

import logging as __logging
import os

from lightning_utilities.core.imports import package_available

from torchmetrics.__about__ import *  # noqa: F403

_logger = __logging.getLogger("torchmetrics")
_logger.addHandler(__logging.StreamHandler())
_logger.setLevel(__logging.INFO)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

if package_available("numpy"):
    # compatibility for AttributeError: `np.Inf` was removed in the NumPy 2.0 release. Use `np.inf` instead
    import numpy

    numpy.Inf = numpy.inf


if package_available("PIL"):
    import PIL

    if not hasattr(PIL, "PILLOW_VERSION"):
        PIL.PILLOW_VERSION = PIL.__version__

if package_available("scipy"):
    import scipy.signal

    # back compatibility patch due to SMRMpy using scipy.signal.hamming
    if not hasattr(scipy.signal, "hamming"):
        scipy.signal.hamming = scipy.signal.windows.hamming

from torchmetrics import functional  # noqa: E402
from torchmetrics.aggregation import (  # noqa: E402
    CatMetric,
    MaxMetric,
    MeanMetric,
    MinMetric,
    RunningMean,
    RunningSum,
    SumMetric,
)
from torchmetrics.audio._deprecated import _PermutationInvariantTraining as PermutationInvariantTraining  # noqa: E402
from torchmetrics.audio._deprecated import (  # noqa: E402
    _ScaleInvariantSignalDistortionRatio as ScaleInvariantSignalDistortionRatio,
)
from torchmetrics.audio._deprecated import (  # noqa: E402
    _ScaleInvariantSignalNoiseRatio as ScaleInvariantSignalNoiseRatio,
)
from torchmetrics.audio._deprecated import _SignalDistortionRatio as SignalDistortionRatio  # noqa: E402
from torchmetrics.audio._deprecated import _SignalNoiseRatio as SignalNoiseRatio  # noqa: E402
from torchmetrics.classification import (  # noqa: E402
    AUROC,
    ROC,
    Accuracy,
    AveragePrecision,
    CalibrationError,
    CohenKappa,
    ConfusionMatrix,
    ExactMatch,
    F1Score,
    FBetaScore,
    HammingDistance,
    HingeLoss,
    JaccardIndex,
    LogAUC,
    MatthewsCorrCoef,
    NegativePredictiveValue,
    Precision,
    PrecisionAtFixedRecall,
    PrecisionRecallCurve,
    Recall,
    RecallAtFixedPrecision,
    SensitivityAtSpecificity,
    Specificity,
    SpecificityAtSensitivity,
    StatScores,
)
from torchmetrics.collections import MetricCollection  # noqa: E402
from torchmetrics.detection._deprecated import _ModifiedPanopticQuality as ModifiedPanopticQuality  # noqa: E402
from torchmetrics.detection._deprecated import _PanopticQuality as PanopticQuality  # noqa: E402
from torchmetrics.image._deprecated import (  # noqa: E402
    _ErrorRelativeGlobalDimensionlessSynthesis as ErrorRelativeGlobalDimensionlessSynthesis,
)
from torchmetrics.image._deprecated import (  # noqa: E402
    _MultiScaleStructuralSimilarityIndexMeasure as MultiScaleStructuralSimilarityIndexMeasure,
)
from torchmetrics.image._deprecated import _PeakSignalNoiseRatio as PeakSignalNoiseRatio  # noqa: E402
from torchmetrics.image._deprecated import _RelativeAverageSpectralError as RelativeAverageSpectralError  # noqa: E402
from torchmetrics.image._deprecated import (  # noqa: E402
    _RootMeanSquaredErrorUsingSlidingWindow as RootMeanSquaredErrorUsingSlidingWindow,
)
from torchmetrics.image._deprecated import _SpectralAngleMapper as SpectralAngleMapper  # noqa: E402
from torchmetrics.image._deprecated import _SpectralDistortionIndex as SpectralDistortionIndex  # noqa: E402
from torchmetrics.image._deprecated import (  # noqa: E402
    _StructuralSimilarityIndexMeasure as StructuralSimilarityIndexMeasure,
)
from torchmetrics.image._deprecated import _TotalVariation as TotalVariation  # noqa: E402
from torchmetrics.image._deprecated import _UniversalImageQualityIndex as UniversalImageQualityIndex  # noqa: E402
from torchmetrics.metric import Metric  # noqa: E402
from torchmetrics.nominal import (  # noqa: E402
    CramersV,
    FleissKappa,
    PearsonsContingencyCoefficient,
    TheilsU,
    TschuprowsT,
)
from torchmetrics.regression import (  # noqa: E402
    ConcordanceCorrCoef,
    CosineSimilarity,
    CriticalSuccessIndex,
    ExplainedVariance,
    KendallRankCorrCoef,
    KLDivergence,
    LogCoshError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogError,
    MinkowskiDistance,
    NormalizedRootMeanSquaredError,
    PearsonCorrCoef,
    R2Score,
    RelativeSquaredError,
    SpearmanCorrCoef,
    SymmetricMeanAbsolutePercentageError,
    TweedieDevianceScore,
    WeightedMeanAbsolutePercentageError,
)
from torchmetrics.retrieval._deprecated import _RetrievalFallOut as RetrievalFallOut  # noqa: E402
from torchmetrics.retrieval._deprecated import _RetrievalHitRate as RetrievalHitRate  # noqa: E402
from torchmetrics.retrieval._deprecated import _RetrievalMAP as RetrievalMAP  # noqa: E402
from torchmetrics.retrieval._deprecated import _RetrievalMRR as RetrievalMRR  # noqa: E402
from torchmetrics.retrieval._deprecated import _RetrievalNormalizedDCG as RetrievalNormalizedDCG  # noqa: E402
from torchmetrics.retrieval._deprecated import _RetrievalPrecision as RetrievalPrecision  # noqa: E402
from torchmetrics.retrieval._deprecated import (  # noqa: E402
    _RetrievalPrecisionRecallCurve as RetrievalPrecisionRecallCurve,
)
from torchmetrics.retrieval._deprecated import _RetrievalRecall as RetrievalRecall  # noqa: E402
from torchmetrics.retrieval._deprecated import (  # noqa: E402
    _RetrievalRecallAtFixedPrecision as RetrievalRecallAtFixedPrecision,
)
from torchmetrics.retrieval._deprecated import _RetrievalRPrecision as RetrievalRPrecision  # noqa: E402
from torchmetrics.text._deprecated import _BLEUScore as BLEUScore  # noqa: E402
from torchmetrics.text._deprecated import _CharErrorRate as CharErrorRate  # noqa: E402
from torchmetrics.text._deprecated import _CHRFScore as CHRFScore  # noqa: E402
from torchmetrics.text._deprecated import _ExtendedEditDistance as ExtendedEditDistance  # noqa: E402
from torchmetrics.text._deprecated import _MatchErrorRate as MatchErrorRate  # noqa: E402
from torchmetrics.text._deprecated import _Perplexity as Perplexity  # noqa: E402
from torchmetrics.text._deprecated import _SacreBLEUScore as SacreBLEUScore  # noqa: E402
from torchmetrics.text._deprecated import _SQuAD as SQuAD  # noqa: E402
from torchmetrics.text._deprecated import _TranslationEditRate as TranslationEditRate  # noqa: E402
from torchmetrics.text._deprecated import _WordErrorRate as WordErrorRate  # noqa: E402
from torchmetrics.text._deprecated import _WordInfoLost as WordInfoLost  # noqa: E402
from torchmetrics.text._deprecated import _WordInfoPreserved as WordInfoPreserved  # noqa: E402
from torchmetrics.wrappers import (  # noqa: E402
    BootStrapper,
    ClasswiseWrapper,
    MetricTracker,
    MinMaxMetric,
    MultioutputWrapper,
    MultitaskWrapper,
)

__all__ = [
    "AUROC",
    "ROC",
    "Accuracy",
    "AveragePrecision",
    "BLEUScore",
    "BootStrapper",
    "CHRFScore",
    "CalibrationError",
    "CatMetric",
    "CharErrorRate",
    "ClasswiseWrapper",
    "CohenKappa",
    "ConcordanceCorrCoef",
    "ConfusionMatrix",
    "CosineSimilarity",
    "CramersV",
    "CriticalSuccessIndex",
    "ErrorRelativeGlobalDimensionlessSynthesis",
    "ExactMatch",
    "ExplainedVariance",
    "ExtendedEditDistance",
    "F1Score",
    "FBetaScore",
    "FleissKappa",
    "HammingDistance",
    "HingeLoss",
    "JaccardIndex",
    "KLDivergence",
    "KendallRankCorrCoef",
    "LogAUC",
    "LogCoshError",
    "MatchErrorRate",
    "MatthewsCorrCoef",
    "MaxMetric",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanMetric",
    "MeanSquaredError",
    "MeanSquaredLogError",
    "Metric",
    "MetricCollection",
    "MetricTracker",
    "MinMaxMetric",
    "MinMetric",
    "MinkowskiDistance",
    "ModifiedPanopticQuality",
    "MultiScaleStructuralSimilarityIndexMeasure",
    "MultioutputWrapper",
    "MultitaskWrapper",
    "NegativePredictiveValue",
    "NormalizedRootMeanSquaredError",
    "PanopticQuality",
    "PeakSignalNoiseRatio",
    "PearsonCorrCoef",
    "PearsonsContingencyCoefficient",
    "PermutationInvariantTraining",
    "Perplexity",
    "Precision",
    "PrecisionAtFixedRecall",
    "PrecisionRecallCurve",
    "R2Score",
    "Recall",
    "RecallAtFixedPrecision",
    "RelativeAverageSpectralError",
    "RelativeSquaredError",
    "RetrievalFallOut",
    "RetrievalHitRate",
    "RetrievalMAP",
    "RetrievalMRR",
    "RetrievalNormalizedDCG",
    "RetrievalPrecision",
    "RetrievalPrecisionRecallCurve",
    "RetrievalRPrecision",
    "RetrievalRecall",
    "RetrievalRecallAtFixedPrecision",
    "RootMeanSquaredErrorUsingSlidingWindow",
    "RunningMean",
    "RunningSum",
    "SQuAD",
    "SacreBLEUScore",
    "ScaleInvariantSignalDistortionRatio",
    "ScaleInvariantSignalNoiseRatio",
    "SensitivityAtSpecificity",
    "SignalDistortionRatio",
    "SignalNoiseRatio",
    "SpearmanCorrCoef",
    "Specificity",
    "SpecificityAtSensitivity",
    "SpectralAngleMapper",
    "SpectralDistortionIndex",
    "StatScores",
    "StructuralSimilarityIndexMeasure",
    "SumMetric",
    "SymmetricMeanAbsolutePercentageError",
    "TheilsU",
    "TotalVariation",
    "TranslationEditRate",
    "TschuprowsT",
    "TweedieDevianceScore",
    "UniversalImageQualityIndex",
    "WeightedMeanAbsolutePercentageError",
    "WordErrorRate",
    "WordInfoLost",
    "WordInfoPreserved",
    "functional",
]

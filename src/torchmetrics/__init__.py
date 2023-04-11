r"""Root package info."""
import logging as __logging
import os

from torchmetrics.__about__ import *  # noqa: F401, F403

_logger = __logging.getLogger("torchmetrics")
_logger.addHandler(__logging.StreamHandler())
_logger.setLevel(__logging.INFO)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

from torchmetrics import functional  # noqa: E402
from torchmetrics.aggregation import CatMetric, MaxMetric, MeanMetric, MinMetric, SumMetric  # noqa: E402
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
    Dice,
    ExactMatch,
    F1Score,
    FBetaScore,
    HammingDistance,
    HingeLoss,
    JaccardIndex,
    MatthewsCorrCoef,
    Precision,
    PrecisionAtFixedRecall,
    PrecisionRecallCurve,
    Recall,
    RecallAtFixedPrecision,
    Specificity,
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
from torchmetrics.nominal import CramersV  # noqa: E402
from torchmetrics.nominal import PearsonsContingencyCoefficient  # noqa: E402
from torchmetrics.nominal import TheilsU, TschuprowsT  # noqa: E402
from torchmetrics.regression import ConcordanceCorrCoef  # noqa: E402
from torchmetrics.regression import CosineSimilarity  # noqa: E402
from torchmetrics.regression import (  # noqa: E402
    ExplainedVariance,
    KendallRankCorrCoef,
    KLDivergence,
    LogCoshError,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogError,
    MinkowskiDistance,
    PearsonCorrCoef,
    R2Score,
    SpearmanCorrCoef,
    SymmetricMeanAbsolutePercentageError,
    TweedieDevianceScore,
    WeightedMeanAbsolutePercentageError,
)
from torchmetrics.retrieval import RetrievalFallOut  # noqa: E402
from torchmetrics.retrieval import RetrievalHitRate  # noqa: E402
from torchmetrics.retrieval import (  # noqa: E402
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalPrecisionRecallCurve,
    RetrievalRecall,
    RetrievalRecallAtFixedPrecision,
    RetrievalRPrecision,
)
from torchmetrics.text import (  # noqa: E402
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
)
from torchmetrics.wrappers import BootStrapper  # noqa: E402
from torchmetrics.wrappers import ClasswiseWrapper, MetricTracker, MinMaxMetric, MultioutputWrapper  # noqa: E402

__all__ = [
    "functional",
    "Accuracy",
    "AUROC",
    "AveragePrecision",
    "BLEUScore",
    "BootStrapper",
    "CalibrationError",
    "CatMetric",
    "ClasswiseWrapper",
    "CharErrorRate",
    "CHRFScore",
    "ConcordanceCorrCoef",
    "CohenKappa",
    "ConfusionMatrix",
    "CosineSimilarity",
    "CramersV",
    "Dice",
    "TweedieDevianceScore",
    "ErrorRelativeGlobalDimensionlessSynthesis",
    "ExactMatch",
    "ExplainedVariance",
    "ExtendedEditDistance",
    "F1Score",
    "FBetaScore",
    "HammingDistance",
    "HingeLoss",
    "JaccardIndex",
    "KendallRankCorrCoef",
    "KLDivergence",
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
    "ModifiedPanopticQuality",
    "MultioutputWrapper",
    "MultiScaleStructuralSimilarityIndexMeasure",
    "PanopticQuality",
    "PearsonCorrCoef",
    "PearsonsContingencyCoefficient",
    "PermutationInvariantTraining",
    "Perplexity",
    "Precision",
    "PrecisionAtFixedRecall",
    "PrecisionRecallCurve",
    "PeakSignalNoiseRatio",
    "R2Score",
    "Recall",
    "RecallAtFixedPrecision",
    "RelativeAverageSpectralError",
    "RetrievalFallOut",
    "RetrievalHitRate",
    "RetrievalMAP",
    "RetrievalMRR",
    "RetrievalNormalizedDCG",
    "RetrievalPrecision",
    "RetrievalRecall",
    "RetrievalRPrecision",
    "RetrievalPrecisionRecallCurve",
    "RetrievalRecallAtFixedPrecision",
    "ROC",
    "RootMeanSquaredErrorUsingSlidingWindow",
    "SacreBLEUScore",
    "SignalDistortionRatio",
    "ScaleInvariantSignalDistortionRatio",
    "ScaleInvariantSignalNoiseRatio",
    "SignalNoiseRatio",
    "SpearmanCorrCoef",
    "Specificity",
    "SpectralAngleMapper",
    "SpectralDistortionIndex",
    "SQuAD",
    "StructuralSimilarityIndexMeasure",
    "StatScores",
    "SumMetric",
    "SymmetricMeanAbsolutePercentageError",
    "TheilsU",
    "TotalVariation",
    "TranslationEditRate",
    "TschuprowsT",
    "UniversalImageQualityIndex",
    "WeightedMeanAbsolutePercentageError",
    "WordErrorRate",
    "WordInfoLost",
    "WordInfoPreserved",
]

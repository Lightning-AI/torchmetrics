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
from functools import partial
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from torch import tensor

from torchmetrics.aggregation import MaxMetric, MeanMetric, MinMetric, SumMetric
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
    ShortTimeObjectiveIntelligibility,
    SignalDistortionRatio,
    SignalNoiseRatio,
)
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.pit import PermutationInvariantTraining
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryCohenKappa,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryFairness,
    BinaryFBetaScore,
    BinaryHammingDistance,
    BinaryHingeLoss,
    BinaryJaccardIndex,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
    BinaryRecallAtFixedPrecision,
    BinaryROC,
    BinarySpecificity,
    Dice,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassCalibrationError,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
    MulticlassExactMatch,
    MulticlassF1Score,
    MulticlassFBetaScore,
    MulticlassHammingDistance,
    MulticlassHingeLoss,
    MulticlassJaccardIndex,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassRecallAtFixedPrecision,
    MulticlassSpecificity,
    MultilabelAveragePrecision,
    MultilabelConfusionMatrix,
    MultilabelCoverageError,
    MultilabelExactMatch,
    MultilabelF1Score,
    MultilabelFBetaScore,
    MultilabelHammingDistance,
    MultilabelJaccardIndex,
    MultilabelMatthewsCorrCoef,
    MultilabelPrecision,
    MultilabelRankingAveragePrecision,
    MultilabelRankingLoss,
    MultilabelRecall,
    MultilabelRecallAtFixedPrecision,
    MultilabelSpecificity,
)
from torchmetrics.detection import PanopticQuality
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
from torchmetrics.image import (
    ErrorRelativeGlobalDimensionlessSynthesis,
    FrechetInceptionDistance,
    InceptionScore,
    KernelInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    RelativeAverageSpectralError,
    RootMeanSquaredErrorUsingSlidingWindow,
    SpectralAngleMapper,
    SpectralDistortionIndex,
    StructuralSimilarityIndexMeasure,
    TotalVariation,
    UniversalImageQualityIndex,
)
from torchmetrics.nominal import CramersV, PearsonsContingencyCoefficient, TheilsU, TschuprowsT
from torchmetrics.regression import (
    ConcordanceCorrCoef,
    CosineSimilarity,
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
from torchmetrics.retrieval import (
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
)
from torchmetrics.text import (
    CharErrorRate,
    ExtendedEditDistance,
    MatchErrorRate,
    WordErrorRate,
    WordInfoLost,
    WordInfoPreserved,
)

_rand_input = lambda: torch.rand(10)
_binary_randint_input = lambda: torch.randint(2, (10,))
_multiclass_randint_input = lambda: torch.randint(3, (10,))
_multiclass_randn_input = lambda: torch.randn(10, 3).softmax(dim=-1)
_multilabel_rand_input = lambda: torch.rand(10, 3)
_multilabel_randint_input = lambda: torch.randint(2, (10, 3))
_audio_input = lambda: torch.randn(8000)
_image_input = lambda: torch.rand([8, 3, 16, 16])
_panoptic_input = lambda: torch.multinomial(
    torch.tensor([1, 1, 0, 0, 0, 0, 1, 1]).float(), 40, replacement=True
).reshape(1, 5, 4, 2)
_nominal_input = lambda: torch.randint(0, 4, (100,))
_text_input_1 = lambda: ["this is the prediction", "there is an other sample"]
_text_input_2 = lambda: ["this is the reference", "there is another one"]


@pytest.mark.parametrize(
    ("metric_class", "preds", "target"),
    [
        pytest.param(BinaryAccuracy, _rand_input, _binary_randint_input, id="binary accuracy"),
        pytest.param(
            partial(MulticlassAccuracy, num_classes=3),
            _multiclass_randint_input,
            _multiclass_randint_input,
            id="multiclass accuracy",
        ),
        pytest.param(
            partial(MulticlassAccuracy, num_classes=3, average=None),
            _multiclass_randint_input,
            _multiclass_randint_input,
            id="multiclass accuracy and average=None",
        ),
        # AUROC
        pytest.param(
            BinaryAUROC,
            _rand_input,
            _binary_randint_input,
            id="binary auroc",
        ),
        pytest.param(
            partial(MulticlassAUROC, num_classes=3),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass auroc",
        ),
        pytest.param(
            partial(MulticlassAUROC, num_classes=3, average=None),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass auroc and average=None",
        ),
        pytest.param(
            BinaryROC,
            _rand_input,
            _binary_randint_input,
            id="binary roc",
        ),
        pytest.param(
            partial(PearsonsContingencyCoefficient, num_classes=5),
            _nominal_input,
            _nominal_input,
            id="pearson contigency coef",
        ),
        pytest.param(partial(TheilsU, num_classes=5), _nominal_input, _nominal_input, id="theils U"),
        pytest.param(partial(TschuprowsT, num_classes=5), _nominal_input, _nominal_input, id="tschuprows T"),
        pytest.param(partial(CramersV, num_classes=5), _nominal_input, _nominal_input, id="cramers V"),
        pytest.param(
            SpectralDistortionIndex,
            _image_input,
            _image_input,
            id="spectral distortion index",
        ),
        pytest.param(
            ErrorRelativeGlobalDimensionlessSynthesis,
            _image_input,
            _image_input,
            id="error relative global dimensionless synthesis",
        ),
        pytest.param(
            PeakSignalNoiseRatio,
            lambda: torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
            lambda: torch.tensor([[3.0, 2.0], [1.0, 0.0]]),
            id="peak signal noise ratio",
        ),
        pytest.param(
            SpectralAngleMapper,
            _image_input,
            _image_input,
            id="spectral angle mapper",
        ),
        pytest.param(
            StructuralSimilarityIndexMeasure,
            _image_input,
            _image_input,
            id="structural similarity index_measure",
        ),
        pytest.param(
            MultiScaleStructuralSimilarityIndexMeasure,
            lambda: torch.rand(3, 3, 180, 180),
            lambda: torch.rand(3, 3, 180, 180),
            id="multiscale structural similarity index measure",
        ),
        pytest.param(
            UniversalImageQualityIndex,
            _image_input,
            _image_input,
            id="universal image quality index",
        ),
        pytest.param(
            partial(PerceptualEvaluationSpeechQuality, fs=8000, mode="nb"),
            _audio_input,
            _audio_input,
            id="perceptual_evaluation_speech_quality",
        ),
        pytest.param(SignalDistortionRatio, _audio_input, _audio_input, id="signal_distortion_ratio"),
        pytest.param(
            ScaleInvariantSignalDistortionRatio, _rand_input, _rand_input, id="scale_invariant_signal_distortion_ratio"
        ),
        pytest.param(SignalNoiseRatio, _rand_input, _rand_input, id="signal_noise_ratio"),
        pytest.param(ScaleInvariantSignalNoiseRatio, _rand_input, _rand_input, id="scale_invariant_signal_noise_ratio"),
        pytest.param(
            partial(ShortTimeObjectiveIntelligibility, fs=8000, extended=False),
            _audio_input,
            _audio_input,
            id="short_time_objective_intelligibility",
        ),
        pytest.param(
            partial(PermutationInvariantTraining, metric_func=scale_invariant_signal_noise_ratio, eval_func="max"),
            lambda: torch.randn(3, 2, 5),
            lambda: torch.randn(3, 2, 5),
            id="permutation_invariant_training",
        ),
        pytest.param(MeanSquaredError, _rand_input, _rand_input, id="mean squared error"),
        pytest.param(SumMetric, _rand_input, None, id="sum metric"),
        pytest.param(MeanMetric, _rand_input, None, id="mean metric"),
        pytest.param(MinMetric, _rand_input, None, id="min metric"),
        pytest.param(MaxMetric, _rand_input, None, id="min metric"),
        pytest.param(
            MeanAveragePrecision,
            lambda: [
                {"boxes": tensor([[258.0, 41.0, 606.0, 285.0]]), "scores": tensor([0.536]), "labels": tensor([0])}
            ],
            lambda: [{"boxes": tensor([[214.0, 41.0, 562.0, 285.0]]), "labels": tensor([0])}],
            id="mean average precision",
        ),
        pytest.param(
            partial(PanopticQuality, things={0, 1}, stuffs={6, 7}),
            _panoptic_input,
            _panoptic_input,
            id="panoptic quality",
        ),
        pytest.param(BinaryAveragePrecision, _rand_input, _binary_randint_input, id="binary average precision"),
        pytest.param(
            partial(BinaryCalibrationError, n_bins=2, norm="l1"),
            _rand_input,
            _binary_randint_input,
            id="binary calibration error",
        ),
        pytest.param(BinaryCohenKappa, _rand_input, _binary_randint_input, id="binary cohen kappa"),
        pytest.param(
            partial(MulticlassAveragePrecision, num_classes=3),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass average precision",
        ),
        pytest.param(
            partial(MulticlassCalibrationError, num_classes=3, n_bins=3, norm="l1"),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass calibration error",
        ),
        pytest.param(
            partial(MulticlassCohenKappa, num_classes=3),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass cohen kappa",
        ),
        pytest.param(
            partial(MultilabelAveragePrecision, num_labels=3),
            _multilabel_rand_input,
            _multilabel_randint_input,
            id="multilabel average precision",
        ),
        pytest.param(BinarySpecificity, _rand_input, _binary_randint_input, id="binary specificity"),
        pytest.param(
            partial(MulticlassSpecificity, num_classes=3),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass specificity",
        ),
        pytest.param(
            partial(MultilabelSpecificity, num_labels=3),
            _multilabel_rand_input,
            _multilabel_randint_input,
            id="multilabel specificity",
        ),
        pytest.param(
            partial(BinaryRecallAtFixedPrecision, min_precision=0.5),
            _rand_input,
            _binary_randint_input,
            id="binary recall at fixed precision",
        ),
        pytest.param(
            partial(MulticlassRecallAtFixedPrecision, num_classes=3, min_precision=0.5),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass recall at fixed precision",
        ),
        pytest.param(
            partial(MultilabelRecallAtFixedPrecision, num_labels=3, min_precision=0.5),
            _multilabel_rand_input,
            _multilabel_randint_input,
            id="multilabel recall at fixed precision",
        ),
        pytest.param(
            partial(MultilabelCoverageError, num_labels=3),
            _multilabel_rand_input,
            _multilabel_randint_input,
            id="multilabel coverage error",
        ),
        pytest.param(
            partial(MultilabelRankingAveragePrecision, num_labels=3),
            _multilabel_rand_input,
            _multilabel_randint_input,
            id="multilabel ranking average precision",
        ),
        pytest.param(
            partial(MultilabelRankingLoss, num_labels=3),
            _multilabel_rand_input,
            _multilabel_randint_input,
            id="multilabel ranking loss",
        ),
        pytest.param(BinaryPrecision, _rand_input, _binary_randint_input, id="binary precision"),
        pytest.param(
            partial(MulticlassPrecision, num_classes=3),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass precision",
        ),
        pytest.param(
            partial(MultilabelPrecision, num_labels=3),
            _multilabel_rand_input,
            _multilabel_randint_input,
            id="multilabel precision",
        ),
        pytest.param(BinaryRecall, _rand_input, _binary_randint_input, id="binary recall"),
        pytest.param(
            partial(MulticlassRecall, num_classes=3),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass recall",
        ),
        pytest.param(
            partial(MultilabelRecall, num_labels=3),
            _multilabel_rand_input,
            _multilabel_randint_input,
            id="multilabel recall",
        ),
        pytest.param(BinaryMatthewsCorrCoef, _rand_input, _binary_randint_input, id="binary matthews corr coef"),
        pytest.param(
            partial(MulticlassMatthewsCorrCoef, num_classes=3),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass matthews corr coef",
        ),
        pytest.param(
            partial(MultilabelMatthewsCorrCoef, num_labels=3),
            _multilabel_rand_input,
            _multilabel_randint_input,
            id="multilabel matthews corr coef",
        ),
        pytest.param(TotalVariation, _image_input, None, id="total variation"),
        pytest.param(
            RootMeanSquaredErrorUsingSlidingWindow,
            _image_input,
            _image_input,
            id="root mean squared error using sliding window",
        ),
        pytest.param(RelativeAverageSpectralError, _image_input, _image_input, id="relative average spectral error"),
        pytest.param(
            LearnedPerceptualImagePatchSimilarity,
            lambda: torch.rand(10, 3, 100, 100),
            lambda: torch.rand(10, 3, 100, 100),
            id="learned perceptual image patch similarity",
        ),
        pytest.param(ConcordanceCorrCoef, _rand_input, _rand_input, id="concordance corr coef"),
        pytest.param(CosineSimilarity, _rand_input, _rand_input, id="cosine similarity"),
        pytest.param(ExplainedVariance, _rand_input, _rand_input, id="explained variance"),
        pytest.param(KendallRankCorrCoef, _rand_input, _rand_input, id="kendall rank corr coef"),
        pytest.param(
            KLDivergence,
            lambda: torch.randn(10, 3).softmax(dim=-1),
            lambda: torch.randn(10, 3).softmax(dim=-1),
            id="kl divergence",
        ),
        pytest.param(LogCoshError, _rand_input, _rand_input, id="log cosh error"),
        pytest.param(MeanSquaredLogError, _rand_input, _rand_input, id="mean squared log error"),
        pytest.param(MeanAbsoluteError, _rand_input, _rand_input, id="mean absolute error"),
        pytest.param(MeanAbsolutePercentageError, _rand_input, _rand_input, id="mean absolute percentage error"),
        pytest.param(partial(MinkowskiDistance, p=3), _rand_input, _rand_input, id="minkowski distance"),
        pytest.param(PearsonCorrCoef, _rand_input, _rand_input, id="pearson corr coef"),
        pytest.param(R2Score, _rand_input, _rand_input, id="r2 score"),
        pytest.param(SpearmanCorrCoef, _rand_input, _rand_input, id="spearman corr coef"),
        pytest.param(SymmetricMeanAbsolutePercentageError, _rand_input, _rand_input, id="symmetric mape"),
        pytest.param(TweedieDevianceScore, _rand_input, _rand_input, id="tweedie deviance score"),
        pytest.param(WeightedMeanAbsolutePercentageError, _rand_input, _rand_input, id="weighted mape"),
        pytest.param(Dice, _multiclass_randint_input, _multiclass_randint_input, id="dice"),
        pytest.param(
            partial(MulticlassExactMatch, num_classes=3),
            lambda: torch.randint(3, (20, 5)),
            lambda: torch.randint(3, (20, 5)),
            id="multiclass exact match",
        ),
        pytest.param(
            partial(MultilabelExactMatch, num_labels=3),
            lambda: torch.randint(2, (20, 3, 5)),
            lambda: torch.randint(2, (20, 3, 5)),
            id="multilabel exact match",
        ),
        pytest.param(BinaryHammingDistance, _rand_input, _binary_randint_input, id="binary hamming distance"),
        pytest.param(
            partial(MulticlassHammingDistance, num_classes=3),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass hamming distance",
        ),
        pytest.param(
            partial(MultilabelHammingDistance, num_labels=3),
            _multilabel_rand_input,
            _multilabel_randint_input,
            id="multilabel hamming distance",
        ),
        pytest.param(BinaryHingeLoss, _rand_input, _binary_randint_input, id="binary hinge loss"),
        pytest.param(
            partial(MulticlassHingeLoss, num_classes=3),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass hinge loss",
        ),
        pytest.param(BinaryJaccardIndex, _rand_input, _binary_randint_input, id="binary jaccard index"),
        pytest.param(
            partial(MulticlassJaccardIndex, num_classes=3),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass jaccard index",
        ),
        pytest.param(
            partial(MultilabelJaccardIndex, num_labels=3),
            _multilabel_rand_input,
            _multilabel_randint_input,
            id="multilabel jaccard index",
        ),
        pytest.param(BinaryF1Score, _rand_input, _binary_randint_input, id="binary f1 score"),
        pytest.param(partial(BinaryFBetaScore, beta=2.0), _rand_input, _binary_randint_input, id="binary fbeta score"),
        pytest.param(
            partial(MulticlassF1Score, num_classes=3),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass f1 score",
        ),
        pytest.param(
            partial(MulticlassFBetaScore, beta=2.0, num_classes=3),
            _multiclass_randn_input,
            _multiclass_randint_input,
            id="multiclass fbeta score",
        ),
        pytest.param(
            partial(MultilabelF1Score, num_labels=3),
            _multilabel_rand_input,
            _multilabel_randint_input,
            id="multilabel f1 score",
        ),
        pytest.param(
            partial(MultilabelFBetaScore, beta=2.0, num_labels=3),
            _multilabel_rand_input,
            _multilabel_randint_input,
            id="multilabel fbeta score",
        ),
        pytest.param(WordInfoPreserved, _text_input_1, _text_input_2, id="word info preserved"),
        pytest.param(WordInfoLost, _text_input_1, _text_input_2, id="word info lost"),
        pytest.param(WordErrorRate, _text_input_1, _text_input_2, id="word error rate"),
        pytest.param(CharErrorRate, _text_input_1, _text_input_2, id="character error rate"),
        pytest.param(ExtendedEditDistance, _text_input_1, _text_input_2, id="extended edit distance"),
        pytest.param(MatchErrorRate, _text_input_1, _text_input_2, id="match error rate"),
    ],
)
@pytest.mark.parametrize("num_vals", [1, 5])
def test_plot_methods(metric_class: object, preds: Callable, target: Callable, num_vals: int):
    """Test the plot method of metrics that only output a single tensor scalar."""
    metric = metric_class()
    input = (lambda: (preds(),)) if target is None else lambda: (preds(), target())

    if num_vals == 1:
        metric.update(*input())
        fig, ax = metric.plot()
    else:
        vals = []
        for _ in range(num_vals):
            val = metric(*input())
            vals.append(val[0] if isinstance(val, tuple) else val)
        fig, ax = metric.plot(vals)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.parametrize(
    ("metric_class", "preds", "target", "index_0"),
    [
        pytest.param(
            partial(KernelInceptionDistance, feature=64, subsets=3, subset_size=20),
            lambda: torch.randint(0, 200, (30, 3, 299, 299), dtype=torch.uint8),
            lambda: torch.randint(0, 200, (30, 3, 299, 299), dtype=torch.uint8),
            True,
            id="kernel inception distance",
        ),
        pytest.param(
            partial(FrechetInceptionDistance, feature=64),
            lambda: torch.randint(0, 200, (30, 3, 299, 299), dtype=torch.uint8),
            lambda: torch.randint(0, 200, (30, 3, 299, 299), dtype=torch.uint8),
            False,
            id="frechet inception distance",
        ),
        pytest.param(
            partial(InceptionScore, feature=64),
            lambda: torch.randint(0, 255, (30, 3, 299, 299), dtype=torch.uint8),
            None,
            True,
            id="inception score",
        ),
    ],
)
@pytest.mark.parametrize("num_vals", [1, 2])
def test_plot_methods_special_image_metrics(metric_class, preds, target, index_0, num_vals):
    """Test the plot method of metrics that only output a single tensor scalar.

    This takes care of FID, KID and inception score image metrics as these have a slightly different call and update
    signature than other metrics.
    """
    metric = metric_class()

    if num_vals == 1:
        if target is None:
            metric.update(preds())
        else:
            metric.update(preds(), real=True)
            metric.update(target(), real=False)
        fig, ax = metric.plot()
    else:
        vals = []
        for _ in range(num_vals):
            if target is None:
                vals.append(metric(preds())[0])
            else:
                metric.update(preds(), real=True)
                metric.update(target(), real=False)
                vals.append(metric.compute() if not index_0 else metric.compute()[0])
                metric.reset()
        fig, ax = metric.plot(vals)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.parametrize(
    ("metric_class", "preds", "target", "indexes"),
    [
        pytest.param(RetrievalMRR, _rand_input, _binary_randint_input, _binary_randint_input, id="retrieval mrr"),
        pytest.param(
            RetrievalPrecision, _rand_input, _binary_randint_input, _binary_randint_input, id="retrieval precision"
        ),
        pytest.param(
            RetrievalRPrecision, _rand_input, _binary_randint_input, _binary_randint_input, id="retrieval r precision"
        ),
        pytest.param(RetrievalRecall, _rand_input, _binary_randint_input, _binary_randint_input, id="retrieval recall"),
        pytest.param(
            RetrievalFallOut, _rand_input, _binary_randint_input, _binary_randint_input, id="retrieval fallout"
        ),
        pytest.param(
            RetrievalHitRate, _rand_input, _binary_randint_input, _binary_randint_input, id="retrieval hitrate"
        ),
        pytest.param(RetrievalMAP, _rand_input, _binary_randint_input, _binary_randint_input, id="retrieval map"),
        pytest.param(
            RetrievalNormalizedDCG,
            _rand_input,
            _binary_randint_input,
            _binary_randint_input,
            id="retrieval normalized dcg",
        ),
        pytest.param(
            RetrievalRecallAtFixedPrecision,
            _rand_input,
            _binary_randint_input,
            _binary_randint_input,
            id="retrieval recall at fixed precision",
        ),
        pytest.param(
            RetrievalPrecisionRecallCurve,
            _rand_input,
            _binary_randint_input,
            _binary_randint_input,
            id="retrieval precision recall curve",
        ),
        pytest.param(
            partial(BinaryFairness, num_groups=2),
            _rand_input,
            _binary_randint_input,
            lambda: torch.ones(10).long(),
            id="binary fairness",
        ),
    ],
)
@pytest.mark.parametrize("num_vals", [1, 2])
def test_plot_methods_retrieval(metric_class, preds, target, indexes, num_vals):
    """Test the plot method for retrieval metrics by themselves, since retrieval metrics requires an extra argument."""
    if num_vals != 1 and metric_class == RetrievalPrecisionRecallCurve:  # curves does not support multiple step plot
        pytest.skip("curve objects does not support plotting multiple steps")

    metric = metric_class()

    if num_vals == 1:
        metric.update(preds(), target(), indexes())
        fig, ax = metric.plot()
    else:
        vals = []
        for _ in range(num_vals):
            res = metric(preds(), target(), indexes())
            vals.append(res[0] if isinstance(res, tuple) else res)
        fig, ax = metric.plot(vals)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)


@pytest.mark.parametrize(
    ("metric_class", "preds", "target", "labels"),
    [
        pytest.param(
            BinaryConfusionMatrix,
            _rand_input,
            _binary_randint_input,
            ["cat", "dog"],
            id="binary confusion matrix",
        ),
        pytest.param(
            partial(MulticlassConfusionMatrix, num_classes=3),
            _multiclass_randint_input,
            _multiclass_randint_input,
            ["cat", "dog", "bird"],
            id="multiclass confusion matrix",
        ),
        pytest.param(
            partial(MultilabelConfusionMatrix, num_labels=3),
            _multilabel_randint_input,
            _multilabel_randint_input,
            ["cat", "dog", "bird"],
            id="multilabel confusion matrix",
        ),
    ],
)
@pytest.mark.parametrize("use_labels", [False, True])
def test_confusion_matrix_plotter(metric_class, preds, target, labels, use_labels):
    """Test confusion matrix that uses specialized plot function."""
    metric = metric_class()
    metric.update(preds(), target())
    labels = labels if use_labels else None
    fig, axs = metric.plot(add_text=True, labels=labels)
    assert isinstance(fig, plt.Figure)
    cond1 = isinstance(axs, matplotlib.axes.Axes)
    cond2 = isinstance(axs, np.ndarray) and all(isinstance(a, matplotlib.axes.Axes) for a in axs)
    assert cond1 or cond2

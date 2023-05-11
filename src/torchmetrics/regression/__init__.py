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
from torchmetrics.regression.concordance import ConcordanceCorrCoef
from torchmetrics.regression.cosine_similarity import CosineSimilarity
from torchmetrics.regression.explained_variance import ExplainedVariance
from torchmetrics.regression.kendall import KendallRankCorrCoef
from torchmetrics.regression.kl_divergence import KLDivergence
from torchmetrics.regression.log_cosh import LogCoshError
from torchmetrics.regression.log_mse import MeanSquaredLogError
from torchmetrics.regression.mae import MeanAbsoluteError
from torchmetrics.regression.mape import MeanAbsolutePercentageError
from torchmetrics.regression.minkowski import MinkowskiDistance
from torchmetrics.regression.mse import MeanSquaredError
from torchmetrics.regression.pearson import PearsonCorrCoef
from torchmetrics.regression.r2 import R2Score
from torchmetrics.regression.rse import RelativeSquaredError
from torchmetrics.regression.spearman import SpearmanCorrCoef
from torchmetrics.regression.symmetric_mape import SymmetricMeanAbsolutePercentageError
from torchmetrics.regression.tweedie_deviance import TweedieDevianceScore
from torchmetrics.regression.wmape import WeightedMeanAbsolutePercentageError

__all__ = [
    "ConcordanceCorrCoef",
    "CosineSimilarity",
    "ExplainedVariance",
    "KendallRankCorrCoef",
    "KLDivergence",
    "LogCoshError",
    "MeanSquaredLogError",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MinkowskiDistance",
    "MeanSquaredError",
    "PearsonCorrCoef",
    "R2Score",
    "RelativeSquaredError",
    "SpearmanCorrCoef",
    "SymmetricMeanAbsolutePercentageError",
    "TweedieDevianceScore",
    "WeightedMeanAbsolutePercentageError",
]

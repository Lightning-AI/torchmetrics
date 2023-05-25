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

from torchmetrics.functional.regression.concordance import concordance_corrcoef
from torchmetrics.functional.regression.cosine_similarity import cosine_similarity
from torchmetrics.functional.regression.explained_variance import explained_variance
from torchmetrics.functional.regression.kendall import kendall_rank_corrcoef
from torchmetrics.functional.regression.kl_divergence import kl_divergence
from torchmetrics.functional.regression.log_cosh import log_cosh_error
from torchmetrics.functional.regression.log_mse import mean_squared_log_error
from torchmetrics.functional.regression.mae import mean_absolute_error
from torchmetrics.functional.regression.mape import mean_absolute_percentage_error
from torchmetrics.functional.regression.minkowski import minkowski_distance
from torchmetrics.functional.regression.mse import mean_squared_error
from torchmetrics.functional.regression.pearson import pearson_corrcoef
from torchmetrics.functional.regression.r2 import r2_score
from torchmetrics.functional.regression.rse import relative_squared_error
from torchmetrics.functional.regression.spearman import spearman_corrcoef
from torchmetrics.functional.regression.symmetric_mape import symmetric_mean_absolute_percentage_error
from torchmetrics.functional.regression.tweedie_deviance import tweedie_deviance_score
from torchmetrics.functional.regression.wmape import weighted_mean_absolute_percentage_error

__all__ = [
    "concordance_corrcoef",
    "cosine_similarity",
    "explained_variance",
    "kendall_rank_corrcoef",
    "kl_divergence",
    "log_cosh_error",
    "mean_squared_log_error",
    "mean_absolute_error",
    "mean_squared_error",
    "pearson_corrcoef",
    "mean_absolute_percentage_error",
    "mean_absolute_percentage_error",
    "minkowski_distance",
    "r2_score",
    "relative_squared_error",
    "spearman_corrcoef",
    "symmetric_mean_absolute_percentage_error",
    "tweedie_deviance_score",
    "weighted_mean_absolute_percentage_error",
]

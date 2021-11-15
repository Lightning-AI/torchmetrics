# Copyright The PyTorch Lightning team.
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
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from torchmetrics.functional.classification.kl_divergence import _kld_compute, _kld_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat


class CoverageError(Metric):
    def __init__(
        self,
        )

class LabelRankingAveragePrecisionScore(Metric):


class LabelRankingLoss(Metric):
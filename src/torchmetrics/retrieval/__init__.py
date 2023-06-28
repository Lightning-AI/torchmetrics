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
from torchmetrics.retrieval.average_precision import RetrievalMAP
from torchmetrics.retrieval.base import RetrievalMetric
from torchmetrics.retrieval.fall_out import RetrievalFallOut
from torchmetrics.retrieval.hit_rate import RetrievalHitRate
from torchmetrics.retrieval.ndcg import RetrievalNormalizedDCG
from torchmetrics.retrieval.precision import RetrievalPrecision
from torchmetrics.retrieval.precision_recall_curve import RetrievalPrecisionRecallCurve, RetrievalRecallAtFixedPrecision
from torchmetrics.retrieval.r_precision import RetrievalRPrecision
from torchmetrics.retrieval.recall import RetrievalRecall
from torchmetrics.retrieval.reciprocal_rank import RetrievalMRR

__all__ = [
    "RetrievalFallOut",
    "RetrievalHitRate",
    "RetrievalMAP",
    "RetrievalMRR",
    "RetrievalNormalizedDCG",
    "RetrievalPrecision",
    "RetrievalPrecisionRecallCurve",
    "RetrievalRecall",
    "RetrievalRecallAtFixedPrecision",
    "RetrievalRPrecision",
]

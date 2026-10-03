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
import pytest
import torch

from torchmetrics.functional.retrieval import (
    retrieval_average_precision,
    retrieval_precision,
    retrieval_recall,
    retrieval_reciprocal_rank,
)
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalPrecision, RetrievalRecall


@pytest.mark.parametrize(
    ("functional", "metric_class", "kwargs", "expected"),
    [
        (retrieval_average_precision, RetrievalMAP, {}, 5 / 6),
        (retrieval_reciprocal_rank, RetrievalMRR, {}, 1.0),
        (retrieval_precision, RetrievalPrecision, {"top_k": 2}, 0.5),
        (retrieval_recall, RetrievalRecall, {"top_k": 2}, 0.5),
    ],
)
@pytest.mark.parametrize("shift", [0.0, -0.3, -0.4, -0.9, -1.0])
def test_retrieval_metrics_are_invariant_to_score_shifts(functional, metric_class, kwargs, expected, shift):
    """Ranking metrics must use relevant items even when their scores are non-positive."""
    preds = torch.tensor([0.9, 0.5, 0.3, 0.1]) + shift
    target = torch.tensor([1, 0, 1, 0])
    indexes = torch.zeros_like(target)

    torch.testing.assert_close(functional(preds, target, **kwargs), torch.tensor(expected))
    torch.testing.assert_close(metric_class(**kwargs)(preds, target, indexes=indexes), torch.tensor(expected))

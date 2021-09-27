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
import pytest
import torch
from sklearn.metrics import pairwise
from torch import tensor

from torchmetrics.functional import embedding_similarity


@pytest.mark.parametrize("similarity", ["cosine", "dot"])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_against_sklearn(similarity, reduction):
    """Compare PL metrics to sklearn version."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch = torch.randn(5, 10, device=device)  # 100 samples in 10 dimensions

    pl_dist = embedding_similarity(batch, similarity=similarity, reduction=reduction, zero_diagonal=False)

    def sklearn_embedding_distance(batch, similarity, reduction):

        metric_func = {"cosine": pairwise.cosine_similarity, "dot": pairwise.linear_kernel}[similarity]

        dist = metric_func(batch, batch)
        if reduction == "mean":
            return dist.mean(axis=-1)
        if reduction == "sum":
            return dist.sum(axis=-1)
        return dist

    sk_dist = sklearn_embedding_distance(batch.cpu().detach().numpy(), similarity=similarity, reduction=reduction)
    sk_dist = tensor(sk_dist, dtype=torch.float, device=device)

    assert torch.allclose(sk_dist, pl_dist)

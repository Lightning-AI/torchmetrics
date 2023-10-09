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

import numpy as np
import pytest
import torch
from sklearn.metrics.cluster import contingency_matrix as sklearn_contingency_matrix
from sklearn.metrics.cluster import entropy as sklearn_entropy
from sklearn.metrics.cluster import pair_confusion_matrix as sklearn_pair_confusion_matrix
from sklearn.metrics.cluster._supervised import _generalized_average as sklearn_generalized_average
from torchmetrics.functional.clustering.utils import (
    calculate_contingency_matrix,
    calculate_entropy,
    calculate_generalized_mean,
    calculate_pair_cluster_confusion_matrix,
)

from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests.helpers import seed_all

seed_all(42)


NUM_CLASSES = 10

_sklearn_inputs = _Input(
    preds=torch.tensor([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]),
    target=torch.tensor([1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 1, 3, 3, 3, 2, 2]),
)

_single_dim_inputs = _Input(
    preds=torch.randint(high=NUM_CLASSES, size=(BATCH_SIZE,)),
    target=torch.randint(high=NUM_CLASSES, size=(BATCH_SIZE,)),
)

_multi_dim_inputs = _Input(
    preds=torch.randint(high=NUM_CLASSES, size=(BATCH_SIZE, 2)),
    target=torch.randint(high=NUM_CLASSES, size=(BATCH_SIZE, 2)),
)


@pytest.mark.parametrize(
    ("preds", "target"),
    [(_sklearn_inputs.preds, _sklearn_inputs.target), (_single_dim_inputs.preds, _single_dim_inputs.target)],
)
class TestContingencyMatrix:
    """Test calculation of dense and sparse contingency matrices."""

    atol = 1e-8

    @pytest.mark.parametrize("eps", [None, 1e-16])
    def test_contingency_matrix_dense(self, preds, target, eps):
        """Check that dense contingency matrices are calculated correctly."""
        tm_c = calculate_contingency_matrix(preds, target, eps)
        sklearn_c = sklearn_contingency_matrix(target, preds, eps=eps)
        assert np.allclose(tm_c, sklearn_c, atol=self.atol)

    def test_contingency_matrix_sparse(self, preds, target):
        """Check that sparse contingency matrices are calculated correctly."""
        tm_c = calculate_contingency_matrix(preds, target, sparse=True).to_dense().numpy()
        sklearn_c = sklearn_contingency_matrix(target, preds, sparse=True).toarray()
        assert np.allclose(tm_c, sklearn_c, atol=self.atol)


def test_eps_and_sparse_error():
    """Check that contingency matrix is not calculated if `eps` is nonzero and `sparse` is True."""
    with pytest.raises(ValueError, match="Cannot specify*"):
        calculate_contingency_matrix(_single_dim_inputs.preds, _single_dim_inputs.target, eps=1e-16, sparse=True)


def test_multidimensional_contingency_error():
    """Check that contingency matrix is not calculated for multidimensional input."""
    with pytest.raises(ValueError, match="Expected 1d*"):
        calculate_contingency_matrix(_multi_dim_inputs.preds, _multi_dim_inputs.target)


@pytest.mark.parametrize("labels", [torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE))])
def test_entropy(labels):
    """Check calculation of entropy."""
    for x in labels:
        assert np.allclose(calculate_entropy(x).numpy(), sklearn_entropy(x))


@pytest.mark.parametrize("labels", [torch.rand(NUM_BATCHES, 2) + 1e-8])
@pytest.mark.parametrize("p", ["min", "geometric", "arithmetic", "max"])
def test_generalized_mean(labels, p):
    """Check calculation of generalized mean for vectors of length 2."""
    for x in labels:
        print(x)
        assert np.allclose(calculate_generalized_mean(x, p), sklearn_generalized_average(x[0], x[1], average_method=p))


@pytest.mark.parametrize(
    "preds, target",
    [(_sklearn_inputs.preds, _sklearn_inputs.target), (_single_dim_inputs.preds, _single_dim_inputs.target)],
)
class TestPairClusterConfusionMatrix:
    """Test that implementation matches sklearns."""

    atol = 1e-8

    def test_pair_cluster_confusion_matrix(self, preds, target):
        """Check that pair cluster confusion matrix is calculated correctly."""
        tm_res = calculate_pair_cluster_confusion_matrix(preds, target)
        sklearn_res = sklearn_pair_confusion_matrix(preds, target)
        assert np.allclose(tm_res, sklearn_res, atol=self.atol)

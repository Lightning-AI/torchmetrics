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
"""The data-based clustering metrics must accumulate at the precision they were handed.

`davies_bouldin_score` and `calinski_harabasz_score` used to allocate their buffers without a ``dtype``, so a
``float64`` input was accumulated at ``float32``. For Davies-Bouldin that is not merely cosmetic: the centroid buffer
rounds each stored centre to roughly seven significant digits, so ``cluster_k - centroids[k]`` cancels catastrophically
once the data sits away from the origin. See https://github.com/Lightning-AI/torchmetrics/issues/3467.

"""

import pytest
import torch

from torchmetrics.clustering import CalinskiHarabaszScore, DaviesBouldinScore
from torchmetrics.functional.clustering import calinski_harabasz_score, davies_bouldin_score, dunn_index

NUM_SAMPLES, NUM_FEATURES, NUM_CLUSTERS = 300, 3, 4

DATA_METRICS = [
    pytest.param(davies_bouldin_score, id="davies_bouldin_score"),
    pytest.param(calinski_harabasz_score, id="calinski_harabasz_score"),
    pytest.param(dunn_index, id="dunn_index"),
]


def _data(dtype: torch.dtype, offset: float = 0.0):
    """Reproducible clustered data, optionally shifted away from the origin."""
    generator = torch.Generator().manual_seed(42)
    data = torch.rand(NUM_SAMPLES, NUM_FEATURES, generator=generator, dtype=torch.float64) + offset
    labels = torch.arange(NUM_SAMPLES) % NUM_CLUSTERS
    return data.to(dtype), labels


def _reference_davies_bouldin(data, labels):
    """The same algorithm, with every buffer explicitly in the data's dtype."""
    unique = labels.unique()
    n = len(unique)
    intra = torch.zeros(n, dtype=data.dtype)
    centroids = torch.zeros((n, data.shape[1]), dtype=data.dtype)
    for i, k in enumerate(unique):
        cluster = data[labels == k]
        centroids[i] = cluster.mean(dim=0)
        intra[i] = (cluster - centroids[i]).pow(2.0).sum(dim=1).sqrt().mean()
    centroid_distances = torch.cdist(centroids, centroids, p=2.0)
    centroid_distances[centroid_distances == 0] = float("inf")
    combined = intra.unsqueeze(0) + intra.unsqueeze(1)
    return (combined / centroid_distances).max(dim=1).values.mean()


@pytest.mark.parametrize("metric", DATA_METRICS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_data_clustering_metric_preserves_dtype(metric, dtype):
    """A float64 input must not come back as float32."""
    data, labels = _data(dtype)
    assert metric(data, labels).dtype == dtype


# subtracting a centroid from off-origin data cancels leading digits, so exact translation invariance is not
# achievable even in float64 -- the tolerance tracks that limit rather than pretending it is zero
TRANSLATIONS = [
    pytest.param(0.0, 1e-12, id="origin"),
    pytest.param(1e3, 1e-10, id="1e3"),
    pytest.param(1e5, 1e-8, id="1e5"),
    pytest.param(1e7, 1e-6, id="1e7"),
]


@pytest.mark.parametrize(("offset", "rtol"), TRANSLATIONS)
def test_davies_bouldin_is_translation_invariant(offset, rtol):
    """Davies-Bouldin depends only on relative geometry, so shifting the data must not change the score.

    This is the regression that matters: with a float32 centroid buffer the score at ``offset=1e7`` came back as
    ``1.73`` where the correct value is ``27.01`` -- a 94% error, orders of magnitude outside these tolerances.

    """
    data, labels = _data(torch.float64, offset=offset)
    at_origin, _ = _data(torch.float64, offset=0.0)
    expected = davies_bouldin_score(at_origin, labels)
    assert torch.allclose(davies_bouldin_score(data, labels), expected, rtol=rtol)


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e5, 1e7])
def test_davies_bouldin_matches_full_precision_reference(offset):
    """Match an independent float64 implementation of the same formula."""
    data, labels = _data(torch.float64, offset=offset)
    assert torch.allclose(davies_bouldin_score(data, labels), _reference_davies_bouldin(data, labels), rtol=1e-12)


@pytest.mark.parametrize(("offset", "rtol"), TRANSLATIONS)
def test_calinski_harabasz_is_translation_invariant(offset, rtol):
    """Calinski-Harabasz is likewise defined by relative geometry only."""
    data, labels = _data(torch.float64, offset=offset)
    at_origin, _ = _data(torch.float64, offset=0.0)
    expected = calinski_harabasz_score(at_origin, labels)
    assert torch.allclose(calinski_harabasz_score(data, labels), expected, rtol=rtol)


def test_data_clustering_metrics_agree_on_dtype():
    """The three data-based clustering metrics should not disagree about what a float64 input deserves."""
    data, labels = _data(torch.float64)
    dtypes = {
        "davies_bouldin": davies_bouldin_score(data, labels).dtype,
        "calinski_harabasz": calinski_harabasz_score(data, labels).dtype,
        "dunn_index": dunn_index(data, labels).dtype,
    }
    assert set(dtypes.values()) == {torch.float64}, dtypes


@pytest.mark.parametrize(
    ("metric_class", "functional"),
    [
        pytest.param(DaviesBouldinScore, davies_bouldin_score, id="davies_bouldin"),
        pytest.param(CalinskiHarabaszScore, calinski_harabasz_score, id="calinski_harabasz"),
    ],
)
def test_class_matches_functional_on_float64(metric_class, functional):
    """The class-based wrapper must not diverge from the functional result."""
    data, labels = _data(torch.float64, offset=1e5)
    metric = metric_class()
    metric.update(data, labels)
    assert torch.allclose(metric.compute().double(), functional(data, labels).double(), rtol=1e-6)


def test_float32_result_is_unchanged():
    """Float32 inputs already allocated float32 buffers, so their values must not move."""
    data, labels = _data(torch.float32)
    assert davies_bouldin_score(data, labels).dtype == torch.float32
    assert calinski_harabasz_score(data, labels).dtype == torch.float32

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
from itertools import combinations

import numpy as np
import pytest
from torchmetrics.clustering.dunn_index import DunnIndex
from torchmetrics.functional.clustering.dunn_index import dunn_index

from unittests.clustering.inputs import (
    _single_target_intrinsic1,
    _single_target_intrinsic2,
)
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)


def _np_dunn_index(data, labels, p):
    unique_labels, inverse_indices = np.unique(labels, return_inverse=True)
    clusters = [data[inverse_indices == label_idx] for label_idx in range(len(unique_labels))]
    centroids = [c.mean(axis=0) for c in clusters]

    intercluster_distance = np.linalg.norm(
        np.stack([a - b for a, b in combinations(centroids, 2)], axis=0), ord=p, axis=1
    )

    max_intracluster_distance = np.stack([
        np.linalg.norm(ci - mu, ord=p, axis=1).max() for ci, mu in zip(clusters, centroids)
    ])

    return intercluster_distance.min() / max_intracluster_distance.max()


@pytest.mark.parametrize(
    "data, labels",
    [
        (_single_target_intrinsic1.data, _single_target_intrinsic1.labels),
        (_single_target_intrinsic2.data, _single_target_intrinsic2.labels),
    ],
)
@pytest.mark.parametrize(
    "p",
    [0, 1, 2],
)
class TestDunnIndex(MetricTester):
    """Test class for `DunnIndex` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_dunn_index(self, data, labels, p, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=data,
            target=labels,
            metric_class=DunnIndex,
            reference_metric=partial(_np_dunn_index, p=p),
            metric_args={"p": p},
        )

    def test_dunn_index_functional(self, data, labels, p):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=data,
            target=labels,
            metric_functional=dunn_index,
            reference_metric=partial(_np_dunn_index, p=p),
            p=p,
        )

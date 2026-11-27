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
from sklearn.metrics import davies_bouldin_score as sklearn_davies_bouldin_score

from torchmetrics.clustering.davies_bouldin_score import DaviesBouldinScore
from torchmetrics.functional.clustering.davies_bouldin_score import davies_bouldin_score
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import _single_target_intrinsic1, _single_target_intrinsic2

seed_all(42)


@pytest.mark.parametrize("offset", [0.0, 1e10, 1e12])
@pytest.mark.parametrize("num_clusters", [2, 26])
def test_davies_bouldin_float64_precision(offset, num_clusters):
    """Double precision centroids must retain separations smaller than float32 resolution."""
    labels = torch.arange(num_clusters).repeat_interleave(3)
    data = (labels.double() * 10 + torch.tensor([0.0, 2.0, 4.0]).repeat(num_clusters)).unsqueeze(1) + offset
    # Each cluster has mean absolute deviation 4/3 and a nearest centroid at distance 10.
    expected = torch.tensor(4 / 15, dtype=torch.float64)
    torch.testing.assert_close(davies_bouldin_score(data, labels), expected)
    metric = DaviesBouldinScore().to(torch.float64)
    for data_batch, labels_batch in zip(data.split(2), labels.split(2)):
        metric.update(data_batch, labels_batch)
    torch.testing.assert_close(metric.compute(), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_davies_bouldin_low_precision_cpu(dtype):
    """Preserving double precision must retain CPU support for low precision inputs."""
    data = torch.tensor([[0.0], [2.0], [4.0], [10.0], [12.0], [14.0]], dtype=dtype)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    expected = torch.tensor(sklearn_davies_bouldin_score(data.float().numpy(), labels.numpy()), dtype=torch.float32)
    torch.testing.assert_close(davies_bouldin_score(data, labels), expected)


@pytest.mark.parametrize(
    ("data", "labels"),
    [
        (_single_target_intrinsic1.data, _single_target_intrinsic1.labels),
        (_single_target_intrinsic2.data, _single_target_intrinsic2.labels),
    ],
)
class TestDaviesBouldinScore(MetricTester):
    """Test class for `DaviesBouldinScore` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_davies_bouldin_score(self, data, labels, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=data,
            target=labels,
            metric_class=DaviesBouldinScore,
            reference_metric=sklearn_davies_bouldin_score,
        )

    def test_davies_bouldin_score_functional(self, data, labels):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=data,
            target=labels,
            metric_functional=davies_bouldin_score,
            reference_metric=sklearn_davies_bouldin_score,
        )

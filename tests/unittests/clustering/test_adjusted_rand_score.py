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
from sklearn.metrics import adjusted_rand_score as sklearn_adjusted_rand_score

from torchmetrics.clustering.adjusted_rand_score import AdjustedRandScore
from torchmetrics.functional.clustering.adjusted_rand_score import adjusted_rand_score
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import _float_inputs_extrinsic, _single_target_extrinsic1, _single_target_extrinsic2


@pytest.mark.parametrize(
    ("preds", "target"),
    [
        (_single_target_extrinsic1.preds, _single_target_extrinsic1.target),
        (_single_target_extrinsic2.preds, _single_target_extrinsic2.target),
    ],
)
class TestAdjustedRandScore(MetricTester):
    """Test class for `AdjustedRandScore` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_adjusted_rand_score(self, preds, target, ddp):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=AdjustedRandScore,
            reference_metric=sklearn_adjusted_rand_score,
        )

    def test_rand_score_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=adjusted_rand_score,
            reference_metric=sklearn_adjusted_rand_score,
        )


def test_rand_score_functional_raises_invalid_task():
    """Check that metric rejects continuous-valued inputs."""
    preds, target = _float_inputs_extrinsic
    with pytest.raises(ValueError, match=r"Expected *"):
        adjusted_rand_score(preds, target)


def test_rand_score_functional_is_symmetric(
    preds=_single_target_extrinsic1.preds, target=_single_target_extrinsic1.target
):
    """Check that the metric functional is symmetric."""
    for p, t in zip(preds, target):
        assert torch.allclose(adjusted_rand_score(p, t), adjusted_rand_score(t, p))


@pytest.mark.parametrize("num_samples", [1000, 100000, 200000])
@pytest.mark.parametrize("overlapping", [False, True])
def test_adjusted_rand_large_sample_counts(num_samples, overlapping):
    """Products of pair counts must not overflow integer arithmetic for large datasets."""
    target = torch.arange(num_samples) % 3
    preds = torch.arange(num_samples) % 5
    if overlapping:
        preds = target.clone()
        preds[::7] = 4
    expected = torch.tensor(sklearn_adjusted_rand_score(preds.numpy(), target.numpy()), dtype=torch.float32)
    torch.testing.assert_close(adjusted_rand_score(preds, target), expected, atol=1e-7, rtol=1e-6)
    metric = AdjustedRandScore()
    for pred_batch, target_batch in zip(preds.split(50000), target.split(50000)):
        metric.update(pred_batch, target_batch)
    torch.testing.assert_close(metric.compute(), expected, atol=1e-7, rtol=1e-6)

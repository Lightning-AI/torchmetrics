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

import pytest
from sklearn.metrics import completeness_score as sklearn_completeness_score
from sklearn.metrics import homogeneity_score as sklearn_homogeneity_score
from sklearn.metrics import v_measure_score as sklearn_v_measure_score

from torchmetrics.clustering.homogeneity_completeness_v_measure import (
    CompletenessScore,
    HomogeneityScore,
    VMeasureScore,
)
from torchmetrics.functional.clustering.homogeneity_completeness_v_measure import (
    completeness_score,
    homogeneity_score,
    v_measure_score,
)
from unittests._helpers import seed_all
from unittests._helpers.testers import MetricTester
from unittests.clustering._inputs import _float_inputs_extrinsic, _single_target_extrinsic1, _single_target_extrinsic2

seed_all(42)


def _reference_sklearn_wrapper(preds, target, fn):
    """Compute reference values using sklearn."""
    return fn(target, preds)


@pytest.mark.parametrize(
    "modular_metric, functional_metric, reference_metric",
    [
        (HomogeneityScore, homogeneity_score, sklearn_homogeneity_score),
        (CompletenessScore, completeness_score, sklearn_completeness_score),
        (VMeasureScore, v_measure_score, sklearn_v_measure_score),
        (
            partial(VMeasureScore, beta=2.0),
            partial(v_measure_score, beta=2.0),
            partial(sklearn_v_measure_score, beta=2.0),
        ),
    ],
)
@pytest.mark.parametrize(
    "preds, target",
    [
        (_single_target_extrinsic1.preds, _single_target_extrinsic1.target),
        (_single_target_extrinsic2.preds, _single_target_extrinsic2.target),
    ],
)
class TestHomogeneityCompletenessVmeasur(MetricTester):
    """Test class for testing homogeneity, completeness and v-measure metrics."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [pytest.param(True, marks=pytest.mark.DDP), False])
    def test_homogeneity_completeness_vmeasure(
        self, modular_metric, functional_metric, reference_metric, preds, target, ddp
    ):
        """Test class implementation of metric."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=modular_metric,
            reference_metric=partial(_reference_sklearn_wrapper, fn=reference_metric),
        )

    def test_homogeneity_completeness_vmeasure_functional(
        self, modular_metric, functional_metric, reference_metric, preds, target
    ):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds=preds,
            target=target,
            metric_functional=functional_metric,
            reference_metric=partial(_reference_sklearn_wrapper, fn=reference_metric),
        )


@pytest.mark.parametrize("functional_metric", [homogeneity_score, completeness_score, v_measure_score])
def test_homogeneity_completeness_vmeasure_functional_raises_invalid_task(functional_metric):
    """Check that metric rejects continuous-valued inputs."""
    preds, target = _float_inputs_extrinsic
    with pytest.raises(ValueError, match=r"Expected *"):
        functional_metric(preds, target)

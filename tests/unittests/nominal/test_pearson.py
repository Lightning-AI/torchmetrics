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
import itertools
import operator

import pandas as pd
import pytest
import torch
from lightning_utilities.core.imports import compare_version
from scipy.stats.contingency import association
from torchmetrics.functional.nominal.pearson import (
    pearsons_contingency_coefficient,
    pearsons_contingency_coefficient_matrix,
)
from torchmetrics.nominal.pearson import PearsonsContingencyCoefficient

from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests.helpers.testers import MetricTester

NUM_CLASSES = 4

_input_default = _Input(
    preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
)

_input_logits = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES), target=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)
)

# No testing with replacing NaN's values is done as not supported in SciPy


@pytest.fixture()
def pearson_matrix_input():
    """Define input in matrix format for the metric."""
    return torch.cat(
        [
            torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES * BATCH_SIZE, 1), dtype=torch.float),
            torch.randint(high=NUM_CLASSES + 2, size=(NUM_BATCHES * BATCH_SIZE, 1), dtype=torch.float),
            torch.randint(high=2, size=(NUM_BATCHES * BATCH_SIZE, 1), dtype=torch.float),
        ],
        dim=-1,
    )


def _pd_pearsons_t(preds, target):
    preds = preds.argmax(1) if preds.ndim == 2 else preds
    target = target.argmax(1) if target.ndim == 2 else target
    preds, target = preds.numpy().astype(int), target.numpy().astype(int)
    observed_values = pd.crosstab(preds, target)

    t = association(observed=observed_values, method="pearson")
    return torch.tensor(t)


def _pd_pearsons_t_matrix(matrix):
    num_variables = matrix.shape[1]
    pearsons_t_matrix_value = torch.ones(num_variables, num_variables)
    for i, j in itertools.combinations(range(num_variables), 2):
        x, y = matrix[:, i], matrix[:, j]
        pearsons_t_matrix_value[i, j] = pearsons_t_matrix_value[j, i] = _pd_pearsons_t(x, y)
    return pearsons_t_matrix_value


@pytest.mark.skipif(compare_version("pandas", operator.lt, "1.3.2"), reason="`dython` package requires `pandas>=1.3.2`")
@pytest.mark.parametrize(
    "preds, target",
    [
        (_input_default.preds, _input_default.target),
        (_input_logits.preds, _input_logits.target),
    ],
)
class TestPearsonsContingencyCoefficient(MetricTester):
    """Test class for `PearsonsContingencyCoefficient` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [False, True])
    def test_pearsons_ta(self, ddp, preds, target):
        """Test class implementation of metric."""
        metric_args = {"num_classes": NUM_CLASSES}
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=PearsonsContingencyCoefficient,
            reference_metric=_pd_pearsons_t,
            metric_args=metric_args,
        )

    def test_pearsons_t_functional(self, preds, target):
        """Test functional implementation of metric."""
        self.run_functional_metric_test(
            preds, target, metric_functional=pearsons_contingency_coefficient, reference_metric=_pd_pearsons_t
        )

    def test_pearsons_t_differentiability(self, preds, target):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        metric_args = {"num_classes": NUM_CLASSES}
        self.run_differentiability_test(
            preds,
            target,
            metric_module=PearsonsContingencyCoefficient,
            metric_functional=pearsons_contingency_coefficient,
            metric_args=metric_args,
        )


@pytest.mark.skipif(compare_version("pandas", operator.lt, "1.3.2"), reason="`dython` package requires `pandas>=1.3.2`")
def test_pearsons_contingency_coefficient_matrix(pearson_matrix_input):
    """Test matrix version of metric works as expected."""
    tm_score = pearsons_contingency_coefficient_matrix(pearson_matrix_input)
    reference_score = _pd_pearsons_t_matrix(pearson_matrix_input)
    assert torch.allclose(tm_score, reference_score)

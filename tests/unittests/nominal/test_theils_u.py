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
from functools import partial

import pytest
import torch
from dython.nominal import theils_u as dython_theils_u
from lightning_utilities.core.imports import compare_version
from torchmetrics.functional.nominal.theils_u import theils_u, theils_u_matrix
from torchmetrics.nominal import TheilsU

from unittests import BATCH_SIZE, NUM_BATCHES, _Input
from unittests.helpers.testers import MetricTester

NUM_CLASSES = 4

_input_default = _Input(
    preds=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
    target=torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE)),
)

# Requires float type to pass NaNs
_preds = torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE), dtype=torch.float)
_preds[0, 0] = float("nan")
_preds[-1, -1] = float("nan")
_target = torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES, BATCH_SIZE), dtype=torch.float)
_target[1, 0] = float("nan")
_target[-1, 0] = float("nan")
_input_with_nans = _Input(preds=_preds, target=_target)

_input_logits = _Input(
    preds=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES), target=torch.rand(NUM_BATCHES, BATCH_SIZE, NUM_CLASSES)
)


@pytest.fixture()
def theils_u_matrix_input():
    """Define input in matrix format for the metric."""
    matrix = torch.cat(
        [
            torch.randint(high=NUM_CLASSES, size=(NUM_BATCHES * BATCH_SIZE, 1), dtype=torch.float),
            torch.randint(high=NUM_CLASSES + 2, size=(NUM_BATCHES * BATCH_SIZE, 1), dtype=torch.float),
            torch.randint(high=2, size=(NUM_BATCHES * BATCH_SIZE, 1), dtype=torch.float),
        ],
        dim=-1,
    )
    matrix[0, 0] = float("nan")
    matrix[-1, -1] = float("nan")
    return matrix


def _dython_theils_u(preds, target, nan_strategy, nan_replace_value):
    preds = preds.argmax(1) if preds.ndim == 2 else preds
    target = target.argmax(1) if target.ndim == 2 else target

    v = dython_theils_u(
        preds.numpy(),
        target.numpy(),
        nan_strategy=nan_strategy,
        nan_replace_value=nan_replace_value,
    )
    return torch.tensor(v)


def _dython_theils_u_matrix(matrix, nan_strategy, nan_replace_value):
    num_variables = matrix.shape[1]
    theils_u_matrix_value = torch.ones(num_variables, num_variables)
    for i, j in itertools.combinations(range(num_variables), 2):
        x, y = matrix[:, i], matrix[:, j]
        theils_u_matrix_value[i, j] = _dython_theils_u(x, y, nan_strategy, nan_replace_value)
        theils_u_matrix_value[j, i] = _dython_theils_u(y, x, nan_strategy, nan_replace_value)
    return theils_u_matrix_value


@pytest.mark.skipif(compare_version("pandas", operator.lt, "1.3.2"), reason="`dython` package requires `pandas>=1.3.2`")
@pytest.mark.parametrize(
    "preds, target",
    [
        (_input_default.preds, _input_default.target),
        (_input_with_nans.preds, _input_with_nans.target),
        (_input_logits.preds, _input_logits.target),
    ],
)
@pytest.mark.parametrize("nan_strategy, nan_replace_value", [("replace", 0.0), ("drop", None)])
class TestTheilsU(MetricTester):
    """Test class for `TheilsU` metric."""

    atol = 1e-5

    @pytest.mark.parametrize("ddp", [False, True])
    def test_theils_u(self, ddp, preds, target, nan_strategy, nan_replace_value):
        """Test class implementation of metric."""
        metric_args = {
            "nan_strategy": nan_strategy,
            "nan_replace_value": nan_replace_value,
            "num_classes": NUM_CLASSES,
        }
        reference_metric = partial(
            _dython_theils_u,
            nan_strategy=nan_strategy,
            nan_replace_value=nan_replace_value,
        )
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            target=target,
            metric_class=TheilsU,
            reference_metric=reference_metric,
            metric_args=metric_args,
        )

    def test_theils_u_functional(self, preds, target, nan_strategy, nan_replace_value):
        """Test functional implementation of metric."""
        metric_args = {
            "nan_strategy": nan_strategy,
            "nan_replace_value": nan_replace_value,
        }
        reference_metric = partial(
            _dython_theils_u,
            nan_strategy=nan_strategy,
            nan_replace_value=nan_replace_value,
        )
        self.run_functional_metric_test(
            preds, target, metric_functional=theils_u, reference_metric=reference_metric, metric_args=metric_args
        )

    def test_theils_u_differentiability(self, preds, target, nan_strategy, nan_replace_value):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        metric_args = {
            "nan_strategy": nan_strategy,
            "nan_replace_value": nan_replace_value,
            "num_classes": NUM_CLASSES,
        }
        self.run_differentiability_test(
            preds,
            target,
            metric_module=TheilsU,
            metric_functional=theils_u,
            metric_args=metric_args,
        )


@pytest.mark.skipif(compare_version("pandas", operator.lt, "1.3.2"), reason="`dython` package requires `pandas>=1.3.2`")
@pytest.mark.parametrize(("nan_strategy", "nan_replace_value"), [("replace", 1.0), ("drop", None)])
def test_theils_u_matrix(theils_u_matrix_input, nan_strategy, nan_replace_value):
    """Test matrix version of metric works as expected."""
    tm_score = theils_u_matrix(theils_u_matrix_input, nan_strategy, nan_replace_value)
    reference_score = _dython_theils_u_matrix(theils_u_matrix_input, nan_strategy, nan_replace_value)
    assert torch.allclose(tm_score, reference_score, atol=1e-6)

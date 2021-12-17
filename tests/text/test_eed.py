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

import numpy as np
import pytest
from torch import Tensor, tensor

from tests.text.helpers import INPUT_ORDER, TextTester
from tests.text.inputs import _inputs_multiple_references, _inputs_single_sentence_multiple_references
from torchmetrics.functional.text.eed import eed
from torchmetrics.text.eed import EED


@pytest.mark.parametrize(
    ["preds", "targets"],
    [(_inputs_multiple_references.preds, _inputs_multiple_references.targets)],
)
class TestEED(TextTester):
    def test_chrf_score_differentiability(self, preds, targets):
        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=EED,
            metric_functional=eed,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )


# test blank edge cases
def test_eed_empty_functional():
    ref = [[]]
    hyp = []
    assert eed(ref, hyp) == tensor(0.0)


def test_eed_empty_class():
    eed_metric = EED()
    ref = [[]]
    hyp = []
    assert eed_metric(ref, hyp) == tensor(0.0)


def test_eed_empty_with_non_empty_hyp_functional():
    ref = [[]]
    hyp = ["python"]
    assert eed(ref, hyp) == tensor(0.0)


def test_eed_empty_with_non_empty_hyp_class():
    eed_metric = EED()
    ref = [[]]
    hyp = ["python"]
    assert eed_metric(ref, hyp) == tensor(0.0)


def test_eed_return_sentence_level_score_functional():
    ref = _inputs_single_sentence_multiple_references.targets
    hyp = _inputs_single_sentence_multiple_references.preds
    _, sentence_eed = eed(ref, hyp, return_sentence_level_score=True)
    isinstance(sentence_eed, Tensor)


def test_eed_return_sentence_level_class():
    metric = EED(return_sentence_level_score=True)
    ref = _inputs_single_sentence_multiple_references.targets
    hyp = _inputs_single_sentence_multiple_references.preds
    _, sentence_eed = metric(ref, hyp)
    isinstance(sentence_eed, Tensor)


# test parallel vs sequential computations
def test_parallelisation_eed():
    references = _inputs_multiple_references.targets
    hypotheses = _inputs_multiple_references.preds

    # batch_size == length of data
    metric = EED()

    sequential_score = metric(references, hypotheses)

    # batch of 1 with compute_on_step == False
    metric = EED(compute_on_step=False)

    for reference, hypothesis in zip(references, hypotheses):
        metric([reference], [hypothesis])

    parallel_score = metric.compute()

    score_comparison = bool(np.isclose(sequential_score, parallel_score))

    assert bool(score_comparison) is True

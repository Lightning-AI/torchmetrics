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
from typing import Any

import pytest
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch import tensor
from torchmetrics.functional.text.bleu import bleu_score
from torchmetrics.text.bleu import BLEUScore

from unittests.text.helpers import TextTester
from unittests.text.inputs import _inputs_multiple_references

# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.SmoothingFunction
smooth_func = SmoothingFunction().method2


def _compute_bleu_metric_nltk(preds, targets, weights, smoothing_function, **kwargs: Any):
    preds_ = [pred.split() for pred in preds]
    targets_ = [[line.split() for line in target] for target in targets]
    return corpus_bleu(
        list_of_references=targets_, hypotheses=preds_, weights=weights, smoothing_function=smoothing_function, **kwargs
    )


@pytest.mark.parametrize(
    ["weights", "n_gram", "smooth_func", "smooth"],
    [
        ([1], 1, None, False),
        ([0.5, 0.5], 2, smooth_func, True),
        ([0.333333, 0.333333, 0.333333], 3, None, False),
        ([0.25, 0.25, 0.25, 0.25], 4, smooth_func, True),
    ],
)
@pytest.mark.parametrize(
    ["preds", "targets"],
    [(_inputs_multiple_references.preds, _inputs_multiple_references.target)],
)
class TestBLEUScore(TextTester):
    """Test class for `BLEUScore` metric."""

    @pytest.mark.parametrize("ddp", [False, True])
    def test_bleu_score_class(self, ddp, preds, targets, weights, n_gram, smooth_func, smooth):
        """Test class implementation of metric."""
        metric_args = {"n_gram": n_gram, "smooth": smooth}
        compute_bleu_metric_nltk = partial(_compute_bleu_metric_nltk, weights=weights, smoothing_function=smooth_func)

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=BLEUScore,
            reference_metric=compute_bleu_metric_nltk,
            metric_args=metric_args,
        )

    def test_bleu_score_functional(self, preds, targets, weights, n_gram, smooth_func, smooth):
        """Test functional implementation of metric."""
        metric_args = {"n_gram": n_gram, "smooth": smooth}
        compute_bleu_metric_nltk = partial(_compute_bleu_metric_nltk, weights=weights, smoothing_function=smooth_func)

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=bleu_score,
            reference_metric=compute_bleu_metric_nltk,
            metric_args=metric_args,
        )

    def test_bleu_score_differentiability(self, preds, targets, weights, n_gram, smooth_func, smooth):
        """Test the differentiability of the metric, according to its `is_differentiable` attribute."""
        metric_args = {"n_gram": n_gram, "smooth": smooth}

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=BLEUScore,
            metric_functional=bleu_score,
            metric_args=metric_args,
        )


def test_bleu_empty_functional():
    """Test that bleu returns 0 when no input is provided."""
    hyp = [[]]
    ref = [[[]]]
    assert bleu_score(hyp, ref) == tensor(0.0)


def test_no_4_gram_functional():
    """Test that bleu returns 0 for 4 gram."""
    preds = ["My full pytorch-lightning"]
    targets = [["My full pytorch-lightning test", "Completely Different"]]
    assert bleu_score(preds, targets) == tensor(0.0)


def test_bleu_empty_class():
    """Test that bleu returns 0 when no input is provided."""
    bleu = BLEUScore()
    preds = [[]]
    targets = [[[]]]
    assert bleu(preds, targets) == tensor(0.0)


def test_no_4_gram_class():
    """Test that bleu returns 0 for 4 gram."""
    bleu = BLEUScore()
    preds = ["My full pytorch-lightning"]
    targets = [["My full pytorch-lightning test", "Completely Different"]]
    assert bleu(preds, targets) == tensor(0.0)


def test_no_and_uniform_weights_functional():
    """Test that implementation works with no weights and uniform weights, and it gives the same result."""
    preds = ["My full pytorch-lightning"]
    targets = [["My full pytorch-lightning test", "Completely Different"]]
    no_weights_score = bleu_score(preds, targets, n_gram=2)
    uniform_weights_score = bleu_score(preds, targets, n_gram=2, weights=[0.5, 0.5])
    assert no_weights_score == uniform_weights_score


def test_no_and_uniform_weights_class():
    """Test that implementation works with no weights and uniform weights, and it gives the same result."""
    no_weights_bleu = BLEUScore(n_gram=2)
    uniform_weights_bleu = BLEUScore(n_gram=2, weights=[0.5, 0.5])

    preds = ["My full pytorch-lightning"]
    targets = [["My full pytorch-lightning test", "Completely Different"]]
    no_weights_score = no_weights_bleu(preds, targets)
    uniform_weights_score = uniform_weights_bleu(preds, targets)
    assert no_weights_score == uniform_weights_score

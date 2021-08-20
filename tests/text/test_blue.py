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

from functools import partial

import pytest
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch import tensor

from tests.text.helpers import INPUT_ORDER, TextTester
from torchmetrics.functional.text.bleu import bleu_score
from torchmetrics.text.bleu import BLEUScore

# example taken from
# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.corpus_bleu
# EXAMPLE 1
HYPOTHESIS_A = tuple(
    "It is a guide to action which ensures that the military always obeys the commands of the party".split()
)
REFERENCE_1A = tuple("It is a guide to action that ensures that the military will forever heed Party commands".split())
REFERENCE_2A = tuple(
    "It is a guiding principle which makes the military forces always being under the command of the Party".split()
)
REFERENCE_3A = tuple("It is the practical guide for the army always to heed the directions of the party".split())

# EXAMPLE 2
HYPOTHESIS_B = tuple("he read the book because he was interested in world history".split())
REFERENCE_1B = tuple("he was interested in world history because he read the book".split())

# EXAMPLE 3
HYPOTHESIS_C = tuple("the cat the cat on the mat".split())
REFERENCE_1C = tuple("the cat is on the mat".split())
REFERENCE_2C = tuple("there is a cat on the mat".split())

TUPLE_OF_REFERENCES = (
    ((REFERENCE_1A, REFERENCE_2A, REFERENCE_3A), tuple([REFERENCE_1B])),
    (tuple([REFERENCE_1B]), (REFERENCE_1C, REFERENCE_2C)),
)
TUPLE_OF_HYPOTHESES = ((HYPOTHESIS_A, HYPOTHESIS_B), (HYPOTHESIS_B, HYPOTHESIS_C))

BATCHES = {"preds": TUPLE_OF_HYPOTHESES, "targets": TUPLE_OF_REFERENCES}

# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.SmoothingFunction
smooth_func = SmoothingFunction().method2


@pytest.mark.parametrize(
    ["weights", "n_gram", "smooth_func", "smooth"],
    [
        pytest.param([1], 1, None, False),
        pytest.param([0.5, 0.5], 2, smooth_func, True),
        pytest.param([0.333333, 0.333333, 0.333333], 3, None, False),
        pytest.param([0.25, 0.25, 0.25, 0.25], 4, smooth_func, True),
    ],
)
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        pytest.param(BATCHES["preds"], BATCHES["targets"]),
    ],
)
class TestBLEUScore(TextTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_bleu_score_class(self, ddp, dist_sync_on_step, preds, targets, weights, n_gram, smooth_func, smooth):
        metric_args = {"n_gram": n_gram, "smooth": smooth}

        nltk_metric = partial(corpus_bleu, weights=weights, smoothing_function=smooth_func)

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=BLEUScore,
            sk_metric=nltk_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )

    def test_bleu_score_functional(self, preds, targets, weights, n_gram, smooth_func, smooth):
        metric_args = {"n_gram": n_gram, "smooth": smooth}
        nltk_metric = partial(corpus_bleu, weights=weights, smoothing_function=smooth_func)

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=bleu_score,
            sk_metric=nltk_metric,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )

    def test_bleu_score_differentiability(self, preds, targets, weights, n_gram, smooth_func, smooth):
        metric_args = {"n_gram": n_gram, "smooth": smooth}

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=BLEUScore,
            metric_functional=bleu_score,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )


def test_bleu_empty_functional():
    hyp = [[]]
    ref = [[[]]]
    assert bleu_score(ref, hyp) == tensor(0.0)


def test_no_4_gram_functional():
    hyps = [["My", "full", "pytorch-lightning"]]
    refs = [[["My", "full", "pytorch-lightning", "test"], ["Completely", "Different"]]]
    assert bleu_score(refs, hyps) == tensor(0.0)


def test_bleu_empty_class():
    bleu = BLEUScore()
    hyp = [[]]
    ref = [[[]]]
    assert bleu(ref, hyp) == tensor(0.0)


def test_no_4_gram_class():
    bleu = BLEUScore()
    hyps = [["My", "full", "pytorch-lightning"]]
    refs = [[["My", "full", "pytorch-lightning", "test"], ["Completely", "Different"]]]
    assert bleu(refs, hyps) == tensor(0.0)

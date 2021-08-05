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

import pytest
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from torch import tensor

from torchmetrics.functional.text.bleu import bleu_score
from torchmetrics.text.bleu import BLEUScore

# example taken from
# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.sentence_bleu
HYPOTHESIS1 = tuple(
    "It is a guide to action which ensures that the military always obeys the commands of the party".split()
)
REFERENCE1 = tuple("It is a guide to action that ensures that the military will forever heed Party commands".split())
REFERENCE2 = tuple(
    "It is a guiding principle which makes the military forces always being under the command of the Party".split()
)
REFERENCE3 = tuple("It is the practical guide for the army always to heed the directions of the party".split())

# example taken from
# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.corpus_bleu
HYP1 = "It is a guide to action which ensures that the military always obeys the commands of the party".split()
HYP2 = "he read the book because he was interested in world history".split()

REF1A = "It is a guide to action that ensures that the military will forever heed Party commands".split()
REF1B = "It is a guiding principle which makes the military force always being under the command of the Party".split()
REF1C = "It is the practical guide for the army always to heed the directions of the party".split()
REF2A = "he was interested in world history because he read the book".split()

TUPLE_OF_REFERENCES = ((REF1A, REF1B, REF1C), tuple([REF2A]))
HYPOTHESES = (HYP1, HYP2)

BATCHES = [
    dict(reference_corpus=[[REF1A, REF1B, REF1C]], translate_corpus=[HYP1]),
    dict(reference_corpus=[[REF2A]], translate_corpus=[HYP2]),
]

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
def test_bleu_score_functional(weights, n_gram, smooth_func, smooth):
    nltk_output = sentence_bleu(
        [REFERENCE1, REFERENCE2, REFERENCE3],
        HYPOTHESIS1,
        weights=weights,
        smoothing_function=smooth_func,
    )
    pl_output = bleu_score([[REFERENCE1, REFERENCE2, REFERENCE3]], [HYPOTHESIS1], n_gram=n_gram, smooth=smooth)
    assert torch.allclose(pl_output, tensor(nltk_output))

    nltk_output = corpus_bleu(TUPLE_OF_REFERENCES, HYPOTHESES, weights=weights, smoothing_function=smooth_func)
    pl_output = bleu_score(TUPLE_OF_REFERENCES, HYPOTHESES, n_gram=n_gram, smooth=smooth)
    assert torch.allclose(pl_output, tensor(nltk_output))


def test_bleu_empty_functional():
    hyp = [[]]
    ref = [[[]]]
    assert bleu_score(ref, hyp) == tensor(0.0)


def test_no_4_gram_functional():
    hyps = [["My", "full", "pytorch-lightning"]]
    refs = [[["My", "full", "pytorch-lightning", "test"], ["Completely", "Different"]]]
    assert bleu_score(refs, hyps) == tensor(0.0)


@pytest.mark.parametrize(
    ["weights", "n_gram", "smooth_func", "smooth"],
    [
        pytest.param([1], 1, None, False),
        pytest.param([0.5, 0.5], 2, smooth_func, True),
        pytest.param([0.333333, 0.333333, 0.333333], 3, None, False),
        pytest.param([0.25, 0.25, 0.25, 0.25], 4, smooth_func, True),
    ],
)
def test_bleu_score_class(weights, n_gram, smooth_func, smooth):
    bleu = BLEUScore(n_gram=n_gram, smooth=smooth)
    nltk_output = sentence_bleu(
        [REFERENCE1, REFERENCE2, REFERENCE3],
        HYPOTHESIS1,
        weights=weights,
        smoothing_function=smooth_func,
    )
    pl_output = bleu([[REFERENCE1, REFERENCE2, REFERENCE3]], [HYPOTHESIS1])
    assert torch.allclose(pl_output, tensor(nltk_output))

    nltk_output = corpus_bleu(TUPLE_OF_REFERENCES, HYPOTHESES, weights=weights, smoothing_function=smooth_func)
    pl_output = bleu(TUPLE_OF_REFERENCES, HYPOTHESES)
    assert torch.allclose(pl_output, tensor(nltk_output))


@pytest.mark.parametrize(
    ["weights", "n_gram", "smooth_func", "smooth"],
    [
        pytest.param([1], 1, None, False),
        pytest.param([0.5, 0.5], 2, smooth_func, True),
        pytest.param([0.333333, 0.333333, 0.333333], 3, None, False),
        pytest.param([0.25, 0.25, 0.25, 0.25], 4, smooth_func, True),
    ],
)
def test_bleu_score_class_batches(weights, n_gram, smooth_func, smooth):
    bleu = BLEUScore(n_gram=n_gram, smooth=smooth)

    nltk_output = corpus_bleu(TUPLE_OF_REFERENCES, HYPOTHESES, weights=weights, smoothing_function=smooth_func)

    for batch in BATCHES:
        bleu.update(batch["reference_corpus"], batch["translate_corpus"])
    pl_output = bleu.compute()
    assert torch.allclose(pl_output, tensor(nltk_output))


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

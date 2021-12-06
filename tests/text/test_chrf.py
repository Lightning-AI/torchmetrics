from functools import partial
from typing import Sequence

import pytest
from torch import Tensor, tensor

from tests.text.helpers import INPUT_ORDER, TextTester
from torchmetrics.functional.text.chrf import chrf_score
from torchmetrics.text.chrf import CHRFScore
from torchmetrics.utilities.imports import _SACREBLEU_AVAILABLE

if _SACREBLEU_AVAILABLE:
    from sacrebleu.metrics import CHRF

# example taken from
# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.corpus_bleu and adjusted
# EXAMPLE 1
HYPOTHESIS_A = "It is a guide to action which ensures that the military always obeys the commands of the party"
REFERENCE_1A = "It is a guide to action that ensures that the military will forever heed Party commands"
REFERENCE_2A = "It is a guiding principle which makes the military forces always being under the command of the Party"

# EXAMPLE 2
HYPOTHESIS_B = "he read the book because he was interested in world history"
REFERENCE_1B = "he was interested in world history because he read the book"
REFERENCE_2B = "It is the practical guide for the army always to heed the directions of the party"

# EXAMPLE 3 (add intentionally whitespaces)
HYPOTHESIS_C = "the cat the   cat on the mat "
REFERENCE_1C = "the  cat is     on the mat "
REFERENCE_2C = "there is a   cat on the mat"

TUPLE_OF_REFERENCES = (
    ((REFERENCE_1A, REFERENCE_2A), (REFERENCE_1B, REFERENCE_2B)),
    ((REFERENCE_1B, REFERENCE_2B), (REFERENCE_1C, REFERENCE_2C)),
)
TUPLE_OF_HYPOTHESES = ((HYPOTHESIS_A, HYPOTHESIS_B), (HYPOTHESIS_B, HYPOTHESIS_C))

BATCHES = {"preds": TUPLE_OF_HYPOTHESES, "targets": TUPLE_OF_REFERENCES}


def sacrebleu_chrf_fn(
    targets: Sequence[Sequence[str]],
    preds: Sequence[str],
    char_order: int,
    word_order: int,
    lowercase: bool,
    whitespace: bool,
) -> Tensor:
    sacrebleu_chrf = CHRF(
        char_order=char_order, word_order=word_order, lowercase=lowercase, whitespace=whitespace, eps_smoothing=True
    )
    # Sacrebleu CHRF expects different format of input
    targets = [[target[i] for target in targets] for i in range(len(targets[0]))]
    sacrebleu_chrf = sacrebleu_chrf.corpus_score(preds, targets).score / 100
    return tensor(sacrebleu_chrf)


@pytest.mark.parametrize(
    ["char_order", "word_order", "lowercase", "whitespace"],
    [
        (6, 2, False, False),
        (6, 2, False, True),
        (4, 2, True, False),
        (6, 0, True, False),
        (6, 0, True, True),
        (4, 0, False, True),
    ],
)
@pytest.mark.parametrize(
    ["preds", "targets"],
    [(BATCHES["preds"], BATCHES["targets"])],
)
@pytest.mark.skipif(not _SACREBLEU_AVAILABLE, reason="test requires sacrebleu")
class TestCHRFScore(TextTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_chrf_score_class(
        self, ddp, dist_sync_on_step, preds, targets, char_order, word_order, lowercase, whitespace
    ):
        metric_args = {
            "n_char_order": char_order,
            "n_word_order": word_order,
            "lowercase": lowercase,
            "whitespace": whitespace,
        }
        nltk_metric = partial(
            sacrebleu_chrf_fn, char_order=char_order, word_order=word_order, lowercase=lowercase, whitespace=whitespace
        )

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=CHRFScore,
            sk_metric=nltk_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )

    def test_chrf_score_functional(self, preds, targets, char_order, word_order, lowercase, whitespace):
        metric_args = {
            "n_char_order": char_order,
            "n_word_order": word_order,
            "lowercase": lowercase,
            "whitespace": whitespace,
        }
        nltk_metric = partial(
            sacrebleu_chrf_fn, char_order=char_order, word_order=word_order, lowercase=lowercase, whitespace=whitespace
        )

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=chrf_score,
            sk_metric=nltk_metric,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )

    def test_chrf_score_differentiability(self, preds, targets, char_order, word_order, lowercase, whitespace):
        metric_args = {
            "n_char_order": char_order,
            "n_word_order": word_order,
            "lowercase": lowercase,
            "whitespace": whitespace,
        }

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=CHRFScore,
            metric_functional=chrf_score,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )


def test_chrf_empty_functional():
    hyp = []
    ref = [[]]
    assert chrf_score(ref, hyp) == tensor(0.0)


def test_chrf_empty_class():
    chrf = CHRFScore()
    hyp = []
    ref = [[]]
    assert chrf(ref, hyp) == tensor(0.0)


def test_chrf_return_sentence_level_score_functional():
    hyp = [HYPOTHESIS_B]
    ref = [[REFERENCE_1B, REFERENCE_2B]]
    _, chrf_sentence_score = chrf_score(ref, hyp, return_sentence_level_score=True)
    isinstance(chrf_sentence_score, Tensor)


def test_chrf_return_sentence_level_class():
    chrf = CHRFScore(return_sentence_level_score=True)
    hyp = [HYPOTHESIS_B]
    ref = [[REFERENCE_1B, REFERENCE_2B]]
    _, chrf_sentence_score = chrf(ref, hyp)
    isinstance(chrf_sentence_score, Tensor)

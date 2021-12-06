from functools import partial
from typing import Sequence

import pytest
from torch import Tensor, tensor

from tests.text.helpers import INPUT_ORDER, TextTester
from torchmetrics.functional.text.ter import ter
from torchmetrics.text.ter import TER
from torchmetrics.utilities.imports import _SACREBLEU_AVAILABLE

if _SACREBLEU_AVAILABLE:
    from sacrebleu.metrics import TER as SacreTER

# example taken from
# https://www.nltk.org/api/nltk.translate.html?highlight=bleu%20score#nltk.translate.bleu_score.corpus_bleu and adjusted
# EXAMPLE 1
HYPOTHESIS_A = "It is a guide to action which ensures that the military always obeys the commands of the party"
REFERENCE_1A = "It is a guide to action that ensures that the military will forever heed Party commands"
REFERENCE_2A = "It is a guiding principle which makes the military forces always being under the command of the Party"

# EXAMPLE 2
HYPOTHESIS_B = "he read The Book because he was interested in World history"
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


def sacrebleu_ter_fn(
    targets: Sequence[Sequence[str]],
    preds: Sequence[str],
    normalized: bool,
    no_punct: bool,
    asian_support: bool,
    case_sensitive: bool,
) -> Tensor:
    sacrebleu_ter = SacreTER(
        normalized=normalized, no_punct=no_punct, asian_support=asian_support, case_sensitive=case_sensitive
    )
    # Sacrebleu CHRF expects different format of input
    targets = [[target[i] for target in targets] for i in range(len(targets[0]))]
    sacrebleu_ter = sacrebleu_ter.corpus_score(preds, targets).score / 100
    return tensor(sacrebleu_ter)


@pytest.mark.parametrize(
    ["normalize", "no_punctuation", "asian_support", "lowercase"],
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
        (True, True, True, True),
    ],
)
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        (BATCHES["preds"], BATCHES["targets"])
    ],
)
@pytest.mark.skipif(not _SACREBLEU_AVAILABLE, reason="test requires sacrebleu")
class TestTER(TextTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_chrf_score_class(
        self, ddp, dist_sync_on_step, preds, targets, normalize, no_punctuation, asian_support, lowercase
    ):
        metric_args = {
            "normalize": normalize,
            "no_punctuation": no_punctuation,
            "asian_support": asian_support,
            "lowercase": lowercase,
        }
        nltk_metric = partial(
            sacrebleu_ter_fn,
            normalized=normalize,
            no_punct=no_punctuation,
            asian_support=asian_support,
            case_sensitive=not lowercase,
        )

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=TER,
            sk_metric=nltk_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )

    def test_ter_score_functional(self, preds, targets, normalize, no_punctuation, asian_support, lowercase):
        metric_args = {
            "normalize": normalize,
            "no_punctuation": no_punctuation,
            "asian_support": asian_support,
            "lowercase": lowercase,
        }
        nltk_metric = partial(
            sacrebleu_ter_fn,
            normalized=normalize,
            no_punct=no_punctuation,
            asian_support=asian_support,
            case_sensitive=not lowercase,
        )

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=ter,
            sk_metric=nltk_metric,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )

    def test_chrf_score_differentiability(self, preds, targets, normalize, no_punctuation, asian_support, lowercase):
        metric_args = {
            "normalize": normalize,
            "no_punctuation": no_punctuation,
            "asian_support": asian_support,
            "lowercase": lowercase,
        }

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=TER,
            metric_functional=ter,
            metric_args=metric_args,
            input_order=INPUT_ORDER.TARGETS_FIRST,
        )


def test_ter_empty_functional():
    hyp = []
    ref = [[]]
    assert ter(ref, hyp) == tensor(0.0)


def test_ter_empty_class():
    ter_metric = TER()
    hyp = []
    ref = [[]]
    assert ter_metric(ref, hyp) == tensor(0.0)


def test_ter_empty_with_non_empty_hyp_functional():
    hyp = ["python"]
    ref = [[]]
    assert ter(ref, hyp) == tensor(0.0)


def test_ter_empty_with_non_empty_hyp_class():
    ter_metric = TER()
    hyp = ["python"]
    ref = [[]]
    assert ter_metric(ref, hyp) == tensor(0.0)


def test_ter_return_sentence_level_score_functional():
    hyp = [HYPOTHESIS_B]
    ref = [[REFERENCE_1B, REFERENCE_2B]]
    _, sentence_ter = ter(ref, hyp, return_sentence_level_score=True)
    isinstance(sentence_ter, Tensor)


def test_ter_return_sentence_level_class():
    ter_metric = TER(return_sentence_level_score=True)
    hyp = [HYPOTHESIS_B]
    ref = [[REFERENCE_1B, REFERENCE_2B]]
    _, sentence_ter = ter_metric(ref, hyp)
    isinstance(sentence_ter, Tensor)

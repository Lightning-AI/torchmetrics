from functools import partial
from typing import Sequence

import pytest
from torch import Tensor, tensor

from tests.text.helpers import TextTester
from tests.text.inputs import _inputs_multiple_references, _inputs_single_sentence_multiple_references
from torchmetrics.functional.text.ter import translation_edit_rate
from torchmetrics.text.ter import TranslationEditRate
from torchmetrics.utilities.imports import _SACREBLEU_AVAILABLE

if _SACREBLEU_AVAILABLE:
    from sacrebleu.metrics import TER as SacreTER


def sacrebleu_ter_fn(
    preds: Sequence[str],
    target: Sequence[Sequence[str]],
    normalized: bool,
    no_punct: bool,
    asian_support: bool,
    case_sensitive: bool,
) -> Tensor:
    sacrebleu_ter = SacreTER(
        normalized=normalized, no_punct=no_punct, asian_support=asian_support, case_sensitive=case_sensitive
    )
    # Sacrebleu CHRF expects different format of input
    target = [[tgt[i] for tgt in target] for i in range(len(target[0]))]
    sacrebleu_ter = sacrebleu_ter.corpus_score(preds, target).score / 100
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
    [(_inputs_multiple_references.preds, _inputs_multiple_references.targets)],
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
            metric_class=TranslationEditRate,
            sk_metric=nltk_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
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
            metric_functional=translation_edit_rate,
            sk_metric=nltk_metric,
            metric_args=metric_args,
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
            metric_module=TranslationEditRate,
            metric_functional=translation_edit_rate,
            metric_args=metric_args,
        )


def test_ter_empty_functional():
    preds = []
    targets = [[]]
    assert translation_edit_rate(preds, targets) == tensor(0.0)


def test_ter_empty_class():
    ter_metric = TranslationEditRate()
    preds = []
    targets = [[]]
    assert ter_metric(preds, targets) == tensor(0.0)


def test_ter_empty_with_non_empty_hyp_functional():
    preds = ["python"]
    targets = [[]]
    assert translation_edit_rate(preds, targets) == tensor(0.0)


def test_ter_empty_with_non_empty_hyp_class():
    ter_metric = TranslationEditRate()
    preds = ["python"]
    targets = [[]]
    assert ter_metric(preds, targets) == tensor(0.0)


def test_ter_return_sentence_level_score_functional():
    preds = _inputs_single_sentence_multiple_references.preds
    targets = _inputs_single_sentence_multiple_references.targets
    _, sentence_ter = translation_edit_rate(preds, targets, return_sentence_level_score=True)
    isinstance(sentence_ter, Tensor)


def test_ter_return_sentence_level_class():
    ter_metric = TranslationEditRate(return_sentence_level_score=True)
    preds = _inputs_single_sentence_multiple_references.preds
    targets = _inputs_single_sentence_multiple_references.targets
    _, sentence_ter = ter_metric(preds, targets)
    isinstance(sentence_ter, Tensor)

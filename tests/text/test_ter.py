from functools import partial
from typing import Sequence

import pytest
from torch import Tensor, tensor

from tests.text.helpers import INPUT_ORDER, TextTester
from tests.text.inputs import _inputs_multiple_references, _inputs_single_sentence_multiple_references
from torchmetrics.functional.text.ter import ter
from torchmetrics.text.ter import TER
from torchmetrics.utilities.imports import _SACREBLEU_AVAILABLE

if _SACREBLEU_AVAILABLE:
    from sacrebleu.metrics import TER as SacreTER


def sacrebleu_ter_fn(
    preds: Sequence[str],
    targets: Sequence[Sequence[str]],
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
            metric_class=TER,
            sk_metric=nltk_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
            input_order=INPUT_ORDER.PREDS_FIRST,
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
            input_order=INPUT_ORDER.PREDS_FIRST,
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
            input_order=INPUT_ORDER.PREDS_FIRST,
        )


def test_ter_empty_functional():
    hyp = []
    ref = [[]]
    assert ter(hyp, ref) == tensor(0.0)


def test_ter_empty_class():
    ter_metric = TER()
    hyp = []
    ref = [[]]
    assert ter_metric(hyp, ref) == tensor(0.0)


def test_ter_empty_with_non_empty_hyp_functional():
    hyp = ["python"]
    ref = [[]]
    assert ter(hyp, ref) == tensor(0.0)


def test_ter_empty_with_non_empty_hyp_class():
    ter_metric = TER()
    hyp = ["python"]
    ref = [[]]
    assert ter_metric(hyp, ref) == tensor(0.0)


def test_ter_return_sentence_level_score_functional():
    hyp = _inputs_single_sentence_multiple_references.preds
    ref = _inputs_single_sentence_multiple_references.targets
    _, sentence_ter = ter(hyp, ref, return_sentence_level_score=True)
    isinstance(sentence_ter, Tensor)


def test_ter_return_sentence_level_class():
    ter_metric = TER(return_sentence_level_score=True)
    hyp = _inputs_single_sentence_multiple_references.preds
    ref = _inputs_single_sentence_multiple_references.targets
    _, sentence_ter = ter_metric(hyp, ref)
    isinstance(sentence_ter, Tensor)

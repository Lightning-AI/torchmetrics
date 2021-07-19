import pytest
from jiwer import compute_measures

from torchmetrics.functional.text.wer import wer
from torchmetrics.text.wer import WER


@pytest.mark.parametrize(
    "hyp,ref,score",
    [(["hello world"], ["hello world"], 0.0), (["hello world"], ["Firwww"], 1.0)],
)
def test_wer_same(hyp, ref, score):
    metric = WER()
    metric.update(hyp, ref)
    assert metric.compute() == score


@pytest.mark.parametrize(
    "hyp,ref,score",
    [(["hello world"], ["hello world"], 0.0), (["hello world"], ["Firwww"], 1.0)],
)
def test_wer_functional(hyp, ref, score):
    assert wer(ref, hyp) == score


@pytest.mark.parametrize(
    "hyp,ref,score",
    [(["hello world"], ["hello world"]), (["hello world"], ["Firwww"])],
)
def test_wer(hyp, ref):
    metric = WER()
    metric.update(hyp, ref)
    assert metric.compute() == compute_measures(ref, hyp)['wer']

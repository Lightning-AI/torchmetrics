import pytest

from torchmetrics.text.wer import WER
from torchmetrics.functional.text.wer import wer


@pytest.mark.parametrize(
    "hyp,ref,score",
    [("hello world", "hello world", 0.0), ("hello world", "Firwww", 1.0)],
)
def test_wer_same(hyp, ref, score):
    metric = WER()
    metric.update(hyp, ref)
    assert metric.compute() == score


@pytest.mark.parametrize(
    "hyp,ref,score",
    [("hello world", "hello world", 0.0), ("hello world", "Firwww", 1.0)],
)
def test_wer_functional(hyp, ref, score):
    assert wer(ref, hyp) == score

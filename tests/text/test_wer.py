import pytest

from torchmetrics.text.wer.wer import WER


def test_wer_same():
    hyp = ["hello world"]
    ref = ["hello world"]
    metric = WER()
    metric.update(hyp, ref)
    assert metric.compute() == 0.0


def test_wer_different():
    hyp = ["firrrrr"]
    ref = ["hello world"]
    metric = WER()
    metric.update(hyp, ref)
    assert metric.compute() == 1.0

import pytest

from torchmetrics.text.wer.wer import WER


@pytest.mark.parametrize(
    "hyp,ref",
    [
        ("hello world", "hello world"),
    ],
)
def test_wer_same(hyp, ref):
    metric = WER()
    metric.update(hyp, ref)
    assert metric.compute() == 0.0


@pytest.mark.parametrize(
    "hyp,ref",
    [("hello world", "Firwww")],
)
def test_wer_different(hyp, ref):
    metric = WER()
    metric.update(hyp, ref)
    assert metric.compute() == 1.0

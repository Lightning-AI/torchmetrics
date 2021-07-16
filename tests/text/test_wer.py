import pytest

from torchmetrics.text.wer.wer import WER


@pytest.mark.parametrize(
    "hyp,ref,score",
    [("hello world", "hello world", 0.0), ("hello world", "Firwww", 1.0)],
)
def test_wer_same(hyp, ref, score):
    metric = WER()
    metric.update(hyp, ref)
    assert metric.compute() == score

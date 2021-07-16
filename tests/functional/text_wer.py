import pytest

from torchmetrics.functional.text.wer import wer


@pytest.mark.parametrize(
    "hyp,ref,score",
    [("hello world", "hello world", 0.0), ("hello world", "Firwww", 1.0)],
)
def test_wer_same(hyp, ref, score):
    assert wer(ref, hyp) == score

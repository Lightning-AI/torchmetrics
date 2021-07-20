import pytest

from torchmetrics.utilities.imports import _JIWER_AVAILABLE

if _JIWER_AVAILABLE:
    from jiwer import compute_measures

from torchmetrics.functional.text.wer import wer
from torchmetrics.text.wer import WER


@pytest.mark.parametrize(
    "hyp,ref,score",
    [(["hello world"], ["hello world"], 0.0), (["hello world"], ["Firwww"], 1.0)],
)
@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
def test_wer_same(hyp, ref, score):
    """
    Test to ensure that the torchmetric WER matches reference scores
    """
    metric = WER()
    metric.update(hyp, ref)
    assert metric.compute() == score


@pytest.mark.parametrize(
    "hyp,ref,score",
    [(["hello world"], ["hello world"], 0.0), (["hello world"], ["Firwww"], 1.0)],
)
@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
def test_wer_functional(hyp, ref, score):
    """
    Test to ensure that the torchmetric functional WER matches the jiwer reference
    """
    assert wer(ref, hyp) == score


@pytest.mark.parametrize(
    "hyp,ref",
    [(["hello world"], ["hello world"]), (["hello world"], ["Firwww"])],
)
@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
def test_wer_reference(hyp, ref):
    """
    Test to ensure that the torchmetric WER matches the jiwer reference
    """
    metric = WER()
    metric.update(hyp, ref)
    assert metric.compute() == compute_measures(ref, hyp)['wer']

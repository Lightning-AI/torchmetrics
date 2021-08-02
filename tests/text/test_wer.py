import pytest

from torchmetrics.utilities.imports import _JIWER_AVAILABLE

if _JIWER_AVAILABLE:
    from jiwer import compute_measures

from torchmetrics.functional.text.wer import wer
from torchmetrics.text.wer import WER


@pytest.mark.parametrize(
    "hyp,ref,score",
    [(["hello world"], ["hello world"], 0.0), (["Firwww"], ["hello world"], 1.0)],
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
    "hyp,ref,expected_score,expected_incorrect,expected_total",
    [
        (["hello world"], ["hello world"], 0.0, 0, 2),
        (["Firwww"], ["hello world"], 1.0, 2, 2),
    ],
)
@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
def test_wer_functional(ref, hyp, expected_score, expected_incorrect, expected_total):
    """
    Test to ensure that the torchmetric functional WER matches the jiwer reference
    """
    assert wer(ref, hyp) == expected_score


@pytest.mark.parametrize(
    "hyp,ref",
    [
        (["hello world"], ["hello world"]),
        (["Firwww"], ["hello world"]),
    ],
)
@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
def test_wer_reference_functional(hyp, ref):
    """
    Test to ensure that the torchmetric functional WER matches the jiwer reference
    """
    assert wer(ref, hyp) == compute_measures(ref, hyp)["wer"]


@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
def test_wer_reference_functional_concatenate():
    """
    Test to ensure that the torchmetric functional WER matches the jiwer reference when concatenating
    """
    ref = ["hello world", "hello world"]
    hyp = ["hello world", "Firwww"]
    assert wer(ref, hyp) == compute_measures(ref, hyp)["wer"]
    assert wer(hyp, ref, concatenate_texts=True) == compute_measures("".join(ref), "".join(hyp))["wer"]


@pytest.mark.parametrize(
    "hyp,ref",
    [
        (["hello world"], ["hello world"]),
        (["Firwww"], ["hello world"]),
    ],
)
@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
def test_wer_reference(hyp, ref):
    """
    Test to ensure that the torchmetric WER matches the jiwer reference
    """
    metric = WER()
    metric.update(hyp, ref)
    assert metric.compute() == compute_measures(ref, hyp)["wer"]


@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
def test_wer_reference_batch():
    """
    Test to ensure that the torchmetric WER matches the jiwer reference with accumulation
    """
    batches = [("hello world", "Firwww"), ("hello world", "hello world")]
    metric = WER()

    for hyp, ref in batches:
        metric.update(ref, hyp)
    reference_score = compute_measures(truth=[x[0] for x in batches], hypothesis=[x[1] for x in batches])["wer"]
    assert metric.compute() == reference_score

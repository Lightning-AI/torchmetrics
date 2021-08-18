import pytest

from torchmetrics.utilities.imports import _JIWER_AVAILABLE

if _JIWER_AVAILABLE:
    from jiwer import compute_measures

from torchmetrics.functional.text.wer import wer
from torchmetrics.text.wer import WER

PREDICTION1 = "hello world"
REFERENCE1 = "hello world"

PREDICTION2 = "what a day"
REFERENCE2 = "what a wonderful day"

BATCH_PREDICTIONS = [PREDICTION1, PREDICTION2]
BATCH_REFERENCES = [REFERENCE1, REFERENCE2]


@pytest.mark.parametrize(
    "prediction,reference",
    [(PREDICTION1, REFERENCE1), (PREDICTION2, REFERENCE2)],
)
def test_wer_functional_single_sentence(prediction, reference):
    """Test functional with strings as inputs."""
    pl_output = wer(prediction, reference)
    jiwer_output = compute_measures(reference, prediction)["wer"]
    assert pl_output == jiwer_output


def test_wer_functional_batch():
    """Test functional with a batch of sentences."""
    pl_output = wer(BATCH_PREDICTIONS, BATCH_REFERENCES)
    jiwer_output = compute_measures(BATCH_REFERENCES, BATCH_PREDICTIONS)["wer"]
    assert pl_output == jiwer_output


@pytest.mark.parametrize(
    "prediction,reference",
    [(PREDICTION1, REFERENCE1), (PREDICTION2, REFERENCE2)],
)
def test_wer_class_single_sentence(prediction, reference):
    """Test class with strings as inputs."""
    metric = WER()
    metric.update(prediction, reference)
    pl_output = metric.compute()
    jiwer_output = compute_measures(reference, prediction)["wer"]
    assert pl_output == jiwer_output


def test_wer_class_batch():
    """Test class with a batch of sentences."""
    metric = WER()
    metric.update(BATCH_PREDICTIONS, BATCH_REFERENCES)
    pl_output = metric.compute()
    jiwer_output = compute_measures(BATCH_REFERENCES, BATCH_PREDICTIONS)["wer"]
    assert pl_output == jiwer_output


def test_wer_class_batches():
    """Test class with two batches of sentences."""
    metric = WER()
    for prediction, reference in zip(BATCH_PREDICTIONS, BATCH_REFERENCES):
        metric.update(prediction, reference)
    pl_output = metric.compute()
    jiwer_output = compute_measures(BATCH_REFERENCES, BATCH_PREDICTIONS)["wer"]
    assert pl_output == jiwer_output

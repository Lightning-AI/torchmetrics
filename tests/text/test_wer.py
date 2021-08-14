import pytest
import torch
from torch import tensor

from torchmetrics.functional.text.wer import wer
from torchmetrics.text.wer import WER

PREDICTION1 = "hello world"
REFERENCE1 = "hello world"
EXPECTED_WER1 = tensor(0.0)

PREDICTION2 = "what a day"
REFERENCE2 = "what a wonderful day"
EXPECTED_WER2 = tensor(0.25)

BATCH_PREDICTIONS = [PREDICTION1, PREDICTION2]
BATCH_REFERENCES = [REFERENCE1, REFERENCE2]
EXPECTED_BATCH_WER = tensor(1 / 6)


@pytest.mark.parametrize(
    "prediction,reference,expected_wer",
    [(PREDICTION1, REFERENCE1, EXPECTED_WER1), (PREDICTION2, REFERENCE2, EXPECTED_WER2)],
)
def test_wer_functional_single_sentence(prediction, reference, expected_wer):
    """Test functional with strings as inputs."""
    pl_output = wer(prediction, reference)
    assert torch.allclose(pl_output, expected_wer)


def test_wer_functional_batch():
    """Test functional with a batch of sentences."""
    pl_output = wer(BATCH_PREDICTIONS, BATCH_REFERENCES)
    assert torch.allclose(pl_output, EXPECTED_BATCH_WER)


@pytest.mark.parametrize(
    "prediction,reference,expected_wer",
    [(PREDICTION1, REFERENCE1, EXPECTED_WER1), (PREDICTION2, REFERENCE2, EXPECTED_WER2)],
)
def test_wer_class_single_sentence(prediction, reference, expected_wer):
    """Test class with strings as inputs."""
    metric = WER()
    metric.update(prediction, reference)
    pl_output = metric.compute()
    assert torch.allclose(pl_output, expected_wer)


def test_wer_class_batch():
    """Test class with a batch of sentences."""
    metric = WER()
    metric.update(BATCH_PREDICTIONS, BATCH_REFERENCES)
    pl_output = metric.compute()
    assert torch.allclose(pl_output, EXPECTED_BATCH_WER)


def test_wer_class_batches():
    """Test class with two batches of sentences."""
    metric = WER()
    for prediction, reference in zip(BATCH_PREDICTIONS, BATCH_REFERENCES):
        metric.update(prediction, reference)
    pl_output = metric.compute()
    assert torch.allclose(pl_output, EXPECTED_BATCH_WER)

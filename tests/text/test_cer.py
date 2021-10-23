import pytest

from tests.text.helpers import TextTester
from torchmetrics.functional.text.cer import char_error_rate
from torchmetrics.text.cer import CharErrorRate


def char_error_rate_metric_fn(preds, targets):
    """
        Computes Character Error Rates.
    """
    cer_score = char_error_rate(preds, targets)
    return round(cer_score.item(), 4)


preds = ["A quick brown fo"]
targets = ["A quick brown fox"]


@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        (preds, targets),
    ],
)
class TestCharErrorRate(TextTester):
    @staticmethod
    def test_char_error_rate_functional(self, preds, targets):
        """
        Computes Character Error Rates of a prediction with target texts.
        """

        cer_score = char_error_rate(preds, targets)
        original_score = char_error_rate_metric_fn(preds, targets)
        assert round(cer_score.item(), 4) == original_score

    @staticmethod
    def test_char_error_rate_metric(self, preds, targets):
        """
        Computes Character Error Rates of a prediction with target texts.
        """

        cer_metric = CharErrorRate()
        cer_score = cer_metric(preds, targets)
        original_score = char_error_rate_metric_fn(preds, targets)

        assert round(cer_score.item(), 4) == original_score

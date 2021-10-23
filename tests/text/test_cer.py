from torchmetrics.text.cer import CharErrorRate
from torchmetrics.functional.text.cer import char_error_rate

import torch
import pytest

from tests.text.helpers import TextTester

def char_error_rate_metric_fn(preds, targets):
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
    def test_char_error_rate_functional(self, preds, targets):
        cer_score = char_error_rate(preds, targets)
        original_score = char_error_rate_metric_fn(preds, targets)
        assert round(cer_score.item(), 4) == original_score


    def test_char_error_rate_metric(self, preds, targets):
        cer_metric = CharErrorRate()
        cer_score = cer_metric(preds, targets)
        original_score = char_error_rate_metric_fn(preds, targets)

        assert round(cer_score.item(), 4) == original_score

    

    

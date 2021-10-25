from typing import Callable, List, Union

import pytest

from tests.text.helpers import INPUT_ORDER, TextTester
from torchmetrics.utilities.imports import _JIWER_AVAILABLE
from torchmetrics.functional.text.cer import char_error_rate
from torchmetrics.text.cer import CharErrorRate

if _JIWER_AVAILABLE:
    from jiwer import compute_measures
else:
    compute_measures = Callable

BATCHES_1 = {"preds": [["hello world"], ["what a day"]], "targets": [["hello world"], ["what a wonderful day"]]}

BATCHES_2 = {
    "preds": [
        ["i like python", "what you mean or swallow"],
        ["hello duck", "i like python"],
    ],
    "targets": [
        ["i like monthy python", "what do you mean, african or european swallow"],
        ["hello world", "i like monthy python"],
    ],
}


def compare_fn(prediction: Union[str, List[str]], reference: Union[str, List[str]]):
    """ compute cer as wer where we just split each word by character """
    # we also need to count spaces, so these need to be mapped to some not so common character
    prediction = map(lambda s: s.replace(" ", "@"), prediction)
    reference = map(lambda s: s.replace(" ", "@"), reference)
    # split into individual characters
    prediction = [char for seq in prediction for char in seq]
    reference = [char for seq in reference for char in seq]
    return compute_measures(reference, prediction)["wer"]


@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        pytest.param(BATCHES_1["preds"], BATCHES_1["targets"]),
        pytest.param(BATCHES_2["preds"], BATCHES_2["targets"]),
    ],
)
class TestCharErrorRate(TextTester):
    """ test class for character error rate """
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_cer_class(self, ddp, dist_sync_on_step, preds, targets):
        """ test modular version of cer"""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=CharErrorRate,
            sk_metric=compare_fn,
            dist_sync_on_step=dist_sync_on_step,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

    def test_cer_functional(self, preds, targets):
        """ test functional version of cer """
        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=char_error_rate,
            sk_metric=compare_fn,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )


    def test_cer_differentiability(self, preds, targets):
        """ test differentiability of cer metric """ 
        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=CharErrorRate,
            metric_functional=char_error_rate,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

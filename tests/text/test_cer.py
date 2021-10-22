from typing import Callable

import pytest

from tests.text.helpers import INPUT_ORDER, TextTester
from torchmetrics.functional.text.cer import char_error_rate
from torchmetrics.text.cer import CharErrorRate

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


class TestCER(TextTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_cer_class(self, ddp, dist_sync_on_step, preds, targets):

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=CharErrorRate,
            sk_metric=Callable,
            dist_sync_on_step=dist_sync_on_step,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

    def test_cer_functional(self, preds, targets):

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=char_error_rate,
            sk_metric=Callable,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

    def test_cer_differentiability(self, preds, targets):

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=CharErrorRate,
            metric_functional=char_error_rate,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

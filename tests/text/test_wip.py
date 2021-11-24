from typing import Callable, List, Union

import pytest
from jiwer import wip

from tests.text.helpers import INPUT_ORDER, TextTester
from torchmetrics.functional.text.wip import word_information_preserved
from torchmetrics.text.wip import WordInfoPreserved
from torchmetrics.utilities.imports import _JIWER_AVAILABLE

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


def _compute_wip_metric_jiwer(prediction: Union[str, List[str]], reference: Union[str, List[str]]):
    return wip(reference, prediction)


@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        pytest.param(BATCHES_1["preds"], BATCHES_1["targets"]),
        pytest.param(BATCHES_2["preds"], BATCHES_2["targets"]),
    ],
)
class TestWordInfoPreserved(TextTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_wip_class(self, ddp, dist_sync_on_step, preds, targets):

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=WordInfoPreserved,
            sk_metric=_compute_wip_metric_jiwer,
            dist_sync_on_step=dist_sync_on_step,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

    def test_wip_functional(self, preds, targets):

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=word_information_preserved,
            sk_metric=_compute_wip_metric_jiwer,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

    def test_wip_differentiability(self, preds, targets):

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=WordInfoPreserved,
            metric_functional=word_information_preserved,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

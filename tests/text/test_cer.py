from typing import Callable, List, Union

import pytest

from tests.text.helpers import INPUT_ORDER, TextTester
from torchmetrics.utilities.imports import _JIWER_AVAILABLE

if _JIWER_AVAILABLE:
    from jiwer import compute_measures
else:
    compute_measures = Callable

from torchmetrics.functional.text.cer import char_error_rate
from torchmetrics.text.cer import CharErrorRate

BATCHES = {
    "preds": [
        ["A", " ", "q", "u", "i", "c", "k", " ", "b", "r", "o", "w", "n", " ", "f", "o"],
    ],
    "targets": [
        ["A", " ", "q", "u", "i", "c", "k", " ", "b", "r", "o", "w", "n", " ", "f", "o", "x"],
    ],
}


def _compute_wer_metric_jiwer(prediction: Union[str, List[str]], reference: Union[str, List[str]]):
    return compute_measures(reference, prediction)["wer"]


@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        pytest.param(BATCHES["preds"], BATCHES["targets"]),
    ],
)
class TestWER(TextTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_wer_class(self, ddp, dist_sync_on_step, preds, targets):

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=CharErrorRate,
            sk_metric=_compute_wer_metric_jiwer,
            dist_sync_on_step=dist_sync_on_step,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

    def test_wer_functional(self, preds, targets):

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=char_error_rate,
            sk_metric=_compute_wer_metric_jiwer,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

    def test_wer_differentiability(self, preds, targets):

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=CharErrorRate,
            metric_functional=char_error_rate,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

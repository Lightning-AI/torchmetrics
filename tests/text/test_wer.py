from typing import Callable, List, Union

import pytest

from tests.text.helpers import INPUT_ORDER, TextTester
from torchmetrics.utilities.imports import _JIWER_AVAILABLE

if _JIWER_AVAILABLE:
    from jiwer import compute_measures
else:
    compute_measures = Callable

from torchmetrics.functional.text.wer import wer
from torchmetrics.text.wer import WER

PREDICTION1 = "hello world"
REFERENCE1 = "hello world"

PREDICTION2 = "what a day"
REFERENCE2 = "what a wonderful day"

BATCHES = {"preds": [[PREDICTION1], [PREDICTION2]], "targets": [[REFERENCE1], [REFERENCE2]]}


def _compute_wer_metric_jiwer(prediction: Union[str, List[str]], reference: Union[str, List[str]]):
    return compute_measures(reference, prediction)["wer"]


@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
class TestWER(TextTester):
    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    @pytest.mark.parametrize(
        ["preds", "targets"],
        [
            pytest.param(BATCHES["preds"], BATCHES["targets"]),
        ],
    )
    def test_wer_class(self, ddp, dist_sync_on_step, preds, targets):

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=WER,
            sk_metric=_compute_wer_metric_jiwer,
            dist_sync_on_step=dist_sync_on_step,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

    @pytest.mark.parametrize(
        ["preds", "targets"],
        [
            pytest.param(BATCHES["preds"], BATCHES["targets"]),
        ],
    )
    def test_wer_functional(self, preds, targets):

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=wer,
            sk_metric=_compute_wer_metric_jiwer,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

    def test_wer_differentiability(self):

        self.run_differentiability_test(
            preds=BATCHES["preds"],
            targets=BATCHES["targets"],
            metric_module=WER,
            metric_functional=wer,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

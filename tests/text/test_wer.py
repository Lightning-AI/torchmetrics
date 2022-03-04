from typing import Callable, List, Union

import pytest

from tests.text.helpers import TextTester
from tests.text.inputs import _inputs_error_rate_batch_size_1, _inputs_error_rate_batch_size_2
from torchmetrics.utilities.imports import _JIWER_AVAILABLE

if _JIWER_AVAILABLE:
    from jiwer import compute_measures
else:
    compute_measures: Callable

from torchmetrics.functional.text.wer import word_error_rate
from torchmetrics.text.wer import WordErrorRate


def _compute_wer_metric_jiwer(preds: Union[str, List[str]], target: Union[str, List[str]]):
    return compute_measures(target, preds)["wer"]


@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        (_inputs_error_rate_batch_size_1.preds, _inputs_error_rate_batch_size_1.targets),
        (_inputs_error_rate_batch_size_2.preds, _inputs_error_rate_batch_size_2.targets),
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
            metric_class=WordErrorRate,
            sk_metric=_compute_wer_metric_jiwer,
            dist_sync_on_step=dist_sync_on_step,
        )

    def test_wer_functional(self, preds, targets):

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=word_error_rate,
            sk_metric=_compute_wer_metric_jiwer,
        )

    def test_wer_differentiability(self, preds, targets):

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=WordErrorRate,
            metric_functional=word_error_rate,
        )

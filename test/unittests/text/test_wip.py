from typing import List, Union

import pytest
from jiwer import wip

from torchmetrics.functional.text.wip import word_information_preserved
from torchmetrics.text.wip import WordInfoPreserved
from torchmetrics.utilities.imports import _JIWER_AVAILABLE
from unittests.text.helpers import TextTester
from unittests.text.inputs import _inputs_error_rate_batch_size_1, _inputs_error_rate_batch_size_2


def _compute_wip_metric_jiwer(preds: Union[str, List[str]], target: Union[str, List[str]]):
    return wip(target, preds)


@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        (_inputs_error_rate_batch_size_1.preds, _inputs_error_rate_batch_size_1.targets),
        (_inputs_error_rate_batch_size_2.preds, _inputs_error_rate_batch_size_2.targets),
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
        )

    def test_wip_functional(self, preds, targets):

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=word_information_preserved,
            sk_metric=_compute_wip_metric_jiwer,
        )

    def test_wip_differentiability(self, preds, targets):

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=WordInfoPreserved,
            metric_functional=word_information_preserved,
        )

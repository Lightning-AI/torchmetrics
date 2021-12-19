from typing import Callable, List, Union

import pytest
from pytest_cases import parametrize_with_cases

from tests.helpers.testers import MetricTesterDDPCases
from tests.text.helpers import INPUT_ORDER, TextTester
from tests.text.inputs import _inputs_error_rate_batch_size_1, _inputs_error_rate_batch_size_2
from torchmetrics.functional.text.cer import char_error_rate
from torchmetrics.text.cer import CharErrorRate
from torchmetrics.utilities.imports import _JIWER_AVAILABLE

if _JIWER_AVAILABLE:
    from jiwer import cer

else:
    compute_measures = Callable


def compare_fn(prediction: Union[str, List[str]], reference: Union[str, List[str]]):
    return cer(reference, prediction)


@pytest.mark.skipif(not _JIWER_AVAILABLE, reason="test requires jiwer")
@pytest.mark.parametrize(
    ["preds", "targets"],
    [
        (_inputs_error_rate_batch_size_1.preds, _inputs_error_rate_batch_size_1.targets),
        (_inputs_error_rate_batch_size_2.preds, _inputs_error_rate_batch_size_2.targets),
    ],
)
class TestCharErrorRate(TextTester):
    """test class for character error rate."""

    @parametrize_with_cases("ddp,device", cases=MetricTesterDDPCases, has_tag="strategy")
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_cer_class(self, ddp, dist_sync_on_step, preds, targets, device):
        """test modular version of cer."""
        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=CharErrorRate,
            sk_metric=compare_fn,
            dist_sync_on_step=dist_sync_on_step,
            device=device,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

    @parametrize_with_cases("device", cases=MetricTesterDDPCases, has_tag="device")
    def test_cer_functional(self, preds, targets, device):
        """test functional version of cer."""
        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=char_error_rate,
            sk_metric=compare_fn,
            device=device,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

    def test_cer_differentiability(self, preds, targets):
        """test differentiability of cer metric."""
        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=CharErrorRate,
            metric_functional=char_error_rate,
            input_order=INPUT_ORDER.PREDS_FIRST,
        )

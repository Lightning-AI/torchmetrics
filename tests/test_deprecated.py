import pytest
import torch

from torchmetrics import Accuracy, Metric


def test_compute_on_step():
    with pytest.warns(
        DeprecationWarning, match="Argument `compute_on_step` is deprecated in v0.8 and will be removed in v0.9"
    ):
        Accuracy(compute_on_step=False)  # any metric will raise the warning


def test_warning_on_overriden_update():
    """Test that deprecation error is raised if user tries to overwrite update method."""

    class OldMetricAPI(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("x", torch.tensor(0))

        def update(self, *args, **kwargs):
            self.x += 1

        def compute(self):
            return self.x

    with pytest.warns(
        DeprecationWarning, match="We detected that you have overwritten the ``update`` method, which was.*"
    ):
        OldMetricAPI()


def test_warning_on_overriden_compute():
    """Test that deprecation error is raised if user tries to overwrite compute method."""

    class OldMetricAPI(Metric):
        def __init__(self):
            super().__init__()
            self.add_state("x", torch.tensor(0))

        def update(self, *args, **kwargs):
            self.x += 1

        def compute(self):
            return self.x

    with pytest.warns(
        DeprecationWarning, match="We detected that you have overwritten the ``compute`` method, which was.*"
    ):
        OldMetricAPI()

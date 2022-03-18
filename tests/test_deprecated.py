import pytest

from torchmetrics import Accuracy


def test_compute_on_step():
    with pytest.warns(
        DeprecationWarning, match="Argument `compute_on_step` is deprecated in v0.8 and will be removed in v0.9"
    ):
        Accuracy(compute_on_step=False)  # any metric will raise the warning

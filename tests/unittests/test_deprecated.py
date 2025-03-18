import pytest
import torch

from torchmetrics.classification import Dice
from torchmetrics.functional.classification import dice


def test_deprecated_dice_from_classification():
    """Ensure that the deprecated `dice` metric from classification raises a warning."""
    preds = torch.randn(10, 2)
    target = torch.randint(0, 2, (10,))

    with pytest.deprecated_call(match="The `dice` metrics is being deprecated from the classification subpackage.*"):
        dice(preds, target)

    with pytest.deprecated_call(match="The `dice` metrics is being deprecated from the classification subpackage.*"):
        Dice()

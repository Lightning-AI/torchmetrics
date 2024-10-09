import pytest
import torch
from torchmetrics.classification import Dice
from torchmetrics.functional.classification import dice
from torchmetrics.functional.regression import kl_divergence
from torchmetrics.regression import KLDivergence, R2Score


def test_deprecated_kl_divergence_input_order():
    """Ensure that the deprecated input order for kl_divergence raises a warning."""
    preds = torch.randn(10, 2)
    target = torch.randn(10, 2)

    with pytest.deprecated_call(match="The input order and naming in metric `kl_divergence` is set to be deprecated.*"):
        kl_divergence(preds, target)

    with pytest.deprecated_call(match="The input order and naming in metric `KLDivergence` is set to be deprecated.*"):
        KLDivergence()


def test_deprecated_dice_from_classification():
    """Ensure that the deprecated `dice` metric from classification raises a warning."""
    preds = torch.randn(10, 2)
    target = torch.randint(0, 2, (10,))

    with pytest.deprecated_call(match="The `dice` metrics is being deprecated from the classification subpackage.*"):
        dice(preds, target)

    with pytest.deprecated_call(match="The `dice` metrics is being deprecated from the classification subpackage.*"):
        Dice()


def test_deprecated_r2_score_num_outputs():
    """Ensure that the deprecated num_outputs argument in R2Score raises a warning."""
    with pytest.deprecated_call(match="Argument `num_outputs` in `R2Score` has been deprecated"):
        R2Score(num_outputs=2)

import pytest
import torch
from torchmetrics.functional.regression import kl_divergence
from torchmetrics.regression import KLDivergence


def test_deprecated_kl_divergence_input_order():
    """Ensure that the deprecated input order for kl_divergence raises a warning."""
    preds = torch.randn(10, 2)
    target = torch.randn(10, 2)

    with pytest.deprecated_call(match="The input order and naming in metric `kl_divergence` is set to be deprecated.*"):
        kl_divergence(preds, target)

    with pytest.deprecated_call(match="The input order and naming in metric `KLDivergence` is set to be deprecated.*"):
        KLDivergence()

import numpy as np
from torch import tensor

from torchmetrics.functional.text.perplexity import perplexity
from torchmetrics.text.perplexity import Perplexity


def test_perplexity_class() -> None:
    """test modular version of perplexity."""

    probs = tensor([[0.2, 0.04, 0.8], [0.34, 0.12, 0.56]])
    mask = tensor([[True, True, False], [True, True, True]])

    expected = 7.3522

    metric = Perplexity()
    metric.update(probs, mask)
    actual = metric.compute()

    np.allclose(expected, actual)


def test_perplexity_functional() -> None:
    """test functional version of perplexity."""

    probs = tensor([[0.2, 0.04, 0.8], [0.34, 0.12, 0.56]])
    mask = tensor([[True, True, False], [True, True, True]])

    expected = 7.3522

    actual = perplexity(probs, mask)

    np.allclose(expected, actual)

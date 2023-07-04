__version__ = "1.0.0"
__author__ = "Lightning-AI et al."
__author_email__ = "name@pytorchlightning.ai"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2020-2023, {__author__}."
__homepage__ = "https://github.com/Lightning-AI/torchmetrics"
__docs__ = "PyTorch native Metrics"
__docs_url__ = "https://torchmetrics.readthedocs.io/en/stable/"
__long_doc__ = """
Torchmetrics is a metrics API created for easy metric development and usage in both PyTorch and
[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/). It was originally a part of
Pytorch Lightning, but got split off so users could take advantage of the large collection of metrics
implemented without having to install Pytorch Lightning (even though we would love for you to try it out).
We currently have around 100+ metrics implemented and we continuously are adding more metrics, both within
already covered domains (classification, regression ect.) but also new domains (object detection ect.).
We make sure that all our metrics are rigorously tested such that you can trust them.
"""

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__docs_url__",
    "__homepage__",
    "__license__",
    "__version__",
]

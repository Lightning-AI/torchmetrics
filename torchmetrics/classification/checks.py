from warnings import warn

warn(
    "`torchmetrics.classification.checks` module has been renamed to `torchmetrics.utilities.checks`"
    " since v0.2 and will be removed in v0.3", DeprecationWarning
)

# todo: deprecated, remove in v0.3
from torchmetrics.utilities.checks import _input_format_classification  # noqa: F401 E402

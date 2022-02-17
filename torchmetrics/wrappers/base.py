import warnings

from torchmetrics.metric import Metric


class BaseWrapper(Metric):
    def __init__(self, **kwargs):
        with warnings.catch_warnings():

            warnings.filterwarnings("ignore", category=DeprecationWarning)

        super

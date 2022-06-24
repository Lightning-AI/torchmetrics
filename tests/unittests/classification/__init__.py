from torchmetrics import Metric


class MetricWrapper(Metric):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    def update(self, *args, **kwargs):
        self.metric.update(*args, **kwargs)

    def compute(self, *args, **kwargs):
        return self.metric.compute(*args, **kwargs)

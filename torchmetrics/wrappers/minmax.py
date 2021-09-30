from pytorch_lightning.utilities.distributed import gather_all_tensors

from torchmetrics.metric import Metric


class MaxMetric(Metric):
    """Pytorch-Lightning Metric that tracks the maximum value of a scalar/tensor across an experiment."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("max_val", default=torch.tensor(0))

    def _wrap_compute(self, compute):
        def wrapped_func(*args, **kwargs):
            # return cached value
            if self._computed is not None:
                return self._computed

            dist_sync_fn = self.dist_sync_fn
            if dist_sync_fn is None and torch.distributed.is_available() and torch.distributed.is_initialized():
                # User provided a bool, so we assume DDP if available
                dist_sync_fn = gather_all_tensors

            if self._to_sync and dist_sync_fn is not None:
                self._sync_dist(dist_sync_fn)

            self._computed = compute(*args, **kwargs)
            # removed the auto-reset

            return self._computed

        return wrapped_func

    def update(self, val):
        self.max_val = val if self.max_val < val else self.max_val

    def compute(self):
        return self.max_val

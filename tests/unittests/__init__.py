import os.path

import torch.cuda

from torchmetrics.utilities.imports import _TORCH_GREATER_EQUAL_1_12
from unittests.helpers.testers import BATCH_SIZE, NUM_BATCHES, NUM_PROCESSES, DummyMetric, MetricTester  # noqa: F401

_PATH_TESTS = os.path.dirname(__file__)
_PATH_ROOT = os.path.dirname(_PATH_TESTS)

if torch.cuda.is_available() and _TORCH_GREATER_EQUAL_1_12:
    # before 1.12, tf32 was used by default
    major, _ = torch.cuda.get_device_capability("cuda")
    ampere_or_later = major >= 8  # Ampere and later leverage tensor cores, where this setting becomes useful
    if ampere_or_later:
        # trade-off precision for performance.
        # For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
        torch.set_float32_matmul_precision("medium")

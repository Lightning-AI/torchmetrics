import os.path

import torch

from unittests.helpers.testers import BATCH_SIZE, NUM_BATCHES, NUM_PROCESSES, DummyMetric, MetricTester  # noqa: F401

_PATH_TESTS = os.path.dirname(__file__)
_PATH_ROOT = os.path.dirname(_PATH_TESTS)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

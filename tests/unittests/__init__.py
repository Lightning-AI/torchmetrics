import os.path

import torch

from unittests.conftest import (  # noqa: F401
    BATCH_SIZE,
    EXTRA_DIM,
    NUM_BATCHES,
    NUM_CLASSES,
    NUM_PROCESSES,
    THRESHOLD,
    setup_ddp,
)

_PATH_TESTS = os.path.dirname(__file__)
_PATH_ROOT = os.path.dirname(_PATH_TESTS)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

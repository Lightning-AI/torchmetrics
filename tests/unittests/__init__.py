import os.path

import numpy
import torch

from unittests.conftest import BATCH_SIZE, EXTRA_DIM, NUM_BATCHES, NUM_CLASSES, NUM_PROCESSES, THRESHOLD, setup_ddp

# adding compatibility for numpy >= 1.24
for tp_name, tp_ins in [("object", object), ("bool", bool), ("int", int), ("float", float)]:
    if not hasattr(numpy, tp_name):
        setattr(numpy, tp_name, tp_ins)

_PATH_TESTS = os.path.dirname(__file__)
_PATH_ROOT = os.path.dirname(_PATH_TESTS)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

__all__ = [
    "BATCH_SIZE",
    "EXTRA_DIM",
    "NUM_BATCHES",
    "NUM_CLASSES",
    "NUM_PROCESSES",
    "THRESHOLD",
    "setup_ddp",
]

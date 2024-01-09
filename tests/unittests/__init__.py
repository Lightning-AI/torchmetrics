import os.path
from typing import NamedTuple

import numpy
import torch
from torch import Tensor

from unittests.conftest import (
    BATCH_SIZE,
    EXTRA_DIM,
    NUM_BATCHES,
    NUM_CLASSES,
    NUM_PROCESSES,
    THRESHOLD,
    setup_ddp,
    skip_on_running_out_of_memory,
)

# adding compatibility for numpy >= 1.24
for tp_name, tp_ins in [("object", object), ("bool", bool), ("int", int), ("float", float)]:
    if not hasattr(numpy, tp_name):
        setattr(numpy, tp_name, tp_ins)

_PATH_TESTS = os.path.dirname(__file__)
_PATH_ROOT = os.path.dirname(_PATH_TESTS)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class _Input(NamedTuple):
    preds: Tensor
    target: Tensor


class _GroupInput(NamedTuple):
    preds: Tensor
    target: Tensor
    groups: Tensor


__all__ = [
    "BATCH_SIZE",
    "EXTRA_DIM",
    "_Input",
    "_GroupInput",
    "NUM_BATCHES",
    "NUM_CLASSES",
    "NUM_PROCESSES",
    "THRESHOLD",
    "setup_ddp",
    "skip_on_running_out_of_memory",
]

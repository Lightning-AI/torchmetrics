import os.path
import warnings
from typing import NamedTuple

import numpy
import torch
from cachier import cachier
from torch import Tensor

from unittests.conftest import (
    BATCH_SIZE,
    EXTRA_DIM,
    NUM_BATCHES,
    NUM_CLASSES,
    NUM_PROCESSES,
    THRESHOLD,
    USE_PYTEST_POOL,
    setup_ddp,
)

# adding compatibility for numpy >= 1.24
for tp_name, tp_ins in [("object", object), ("bool", bool), ("int", int), ("float", float)]:
    if not hasattr(numpy, tp_name):
        setattr(numpy, tp_name, tp_ins)

_PATH_UNITTESTS = os.path.dirname(__file__)
_PATH_ALL_TESTS = os.path.dirname(_PATH_UNITTESTS)
_PATH_TEST_CACHE = os.getenv("PYTEST_REFERENCE_CACHE", os.path.join(_PATH_ALL_TESTS, "_cache-references"))


_reference_cachier = cachier(cache_dir=_PATH_TEST_CACHE, separate_files=True)

# ignore FutureWarnings while testing (mainly appearing with DDP runs)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")


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
    "NUM_BATCHES",
    "NUM_CLASSES",
    "NUM_PROCESSES",
    "THRESHOLD",
    "USE_PYTEST_POOL",
    "_GroupInput",
    "_Input",
    "setup_ddp",
]

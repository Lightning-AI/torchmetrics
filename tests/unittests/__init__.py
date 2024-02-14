import functools
import hashlib
import os.path
import pickle
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
    setup_ddp,
)

# adding compatibility for numpy >= 1.24
for tp_name, tp_ins in [("object", object), ("bool", bool), ("int", int), ("float", float)]:
    if not hasattr(numpy, tp_name):
        setattr(numpy, tp_name, tp_ins)

_PATH_UNITTESTS = os.path.dirname(__file__)
_PATH_ALL_TESTS = os.path.dirname(_PATH_UNITTESTS)
_PATH_TEST_CACHE = os.getenv("PYTEST_REFERENCE_CACHE", os.path.join(_PATH_ALL_TESTS, "_reference-cache"))


def _convert_list_to_tuple(val):
    return tuple(_convert_list_to_tuple(x) for x in val) if isinstance(val, list) else val


def _cachier_hash_func(args, kwds):
    # convert lists to tuples to make them hashable
    args = _convert_list_to_tuple(args)
    kwds = {k: _convert_list_to_tuple(v) for k, v in kwds.items()}
    # use pickle to hash the arguments
    key = functools._make_key(args, kwds, typed=True)
    hash_sha256 = hashlib.sha256()
    for item in key:
        hash_sha256.update(pickle.dumps(item))
    return hash_sha256.hexdigest()


_reference_cachier = cachier(cache_dir=_PATH_TEST_CACHE, hash_func=_cachier_hash_func)


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
]

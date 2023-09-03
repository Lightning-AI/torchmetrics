from collections import namedtuple

import torch
from sklearn.datasets import make_blobs

Input = namedtuple("Input", ["x", "labels"])

NUM_BATCHES = 4
NUM_SAMPLES = 50
NUM_FEATURES = 2
NUM_CLASSES = 3


def _batch_blobs(num_batches, num_samples, num_features, num_classes):
    x = []
    labels = []

    for _ in range(num_batches):
        _x, _labels = make_blobs(num_samples, num_features, centers=num_classes)
        x.append(torch.tensor(_x))
        labels.append(torch.tensor(_labels))

    return Input(x=torch.stack(x), labels=torch.stack(labels))


_input_blobs = _batch_blobs(NUM_BATCHES, NUM_SAMPLES, NUM_FEATURES, NUM_CLASSES)

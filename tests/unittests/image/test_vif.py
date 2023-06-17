from collections import namedtuple

import numpy as np
import torch
from sewar.utils import fspecial, Filter
from sewar.full_ref import vifp

from torchmetrics.functional.image.vif import _filter
from unittests import BATCH_SIZE, NUM_BATCHES
from unittests.helpers import seed_all
from unittests.helpers.testers import MetricTester

seed_all(42)

Input = namedtuple("Input", ["preds", "target"])
_input_size = (NUM_BATCHES, BATCH_SIZE, 32, 32)
_inputs = Input(preds=torch.randint(0, 255, _input_size, dtype=torch.float),
                target=torch.randint(0, 255, _input_size, dtype=torch.float))


class TestVIF(MetricTester):

    def test_filter_creation(self):
        for scale in range(1, 5):
            n = 2 ** (4 - scale + 1) + 1
            our_filter = _filter(win_size=n, sigma=n / 5).numpy()
            ref_filter = fspecial(Filter.GAUSSIAN, n, sigma=n / 5)

            diff = np.abs(our_filter - ref_filter)
            assert np.all(diff < 1e-6)

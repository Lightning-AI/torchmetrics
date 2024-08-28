from lightning_utilities.core.imports import package_available

testing = package_available("pytest") and package_available("doctest")

if testing:
    import doctest
    import pytest

    MANUAL_SEED = doctest.register_optionflag("MANUAL_SEED")

    @pytest.fixture(autouse=True)
    def reset_random_seed(seed: int = 42):
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def pytest_collect_file(parent, path):
        if path.ext == ".py":
            return DoctestModule.from_parent(parent, fspath=path)

    class DoctestModule(pytest.Module):
        def collect(self):
            for item in super().collect():
                if isinstance(item, pytest.DoctestItem):
                    item.add_marker(pytest.mark.usefixtures("reset_random_seed"))
                yield item
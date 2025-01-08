from pathlib import Path
from typing import Optional

from lightning_utilities.core.imports import package_available

if package_available("pytest") and package_available("doctest"):
    import doctest

    import pytest

    MANUAL_SEED = doctest.register_optionflag("MANUAL_SEED")

    @pytest.fixture(autouse=True)
    def reset_random_seed(seed: int = 42) -> None:
        """Reset the random seed before running each doctest."""
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    class DoctestModule(pytest.Module):
        """A custom module class that augments collected doctests with the reset_random_seed fixture."""

        def collect(self) -> GeneratorExit:
            """Augment collected doctests with the reset_random_seed fixture."""
            for item in super().collect():
                if isinstance(item, pytest.DoctestItem):
                    item.add_marker(pytest.mark.usefixtures("reset_random_seed"))
                yield item

    def pytest_collect_file(parent: Path, path: Path) -> Optional[DoctestModule]:
        """Collect doctests and add the reset_random_seed fixture."""
        if path.ext == ".py":
            return DoctestModule.from_parent(parent, path=Path(path))
        return None

# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from lightning_utilities import module_available
from torch.utils.data import Dataset

if module_available("lightning"):
    from lightning.pytorch import LightningModule
else:
    from pytorch_lightning import LightningModule


class RandomDictStringDataset(Dataset):
    """Class for creating a dictionary of random strings."""

    def __init__(self, size, length) -> None:
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index) -> dict:
        """Get datapoint."""
        return {"id": str(index), "x": self.data[index]}

    def __len__(self) -> int:
        """Return length of dataset."""
        return self.len


class RandomDataset(Dataset):
    """Random dataset for testing PL Module."""

    def __init__(self, size, length) -> None:
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index) -> torch.Tensor:
        """Get datapoint."""
        return self.data[index]

    def __len__(self) -> int:
        """Get length of dataset."""
        return self.len


class BoringModel(LightningModule):
    """Testing PL Module.

    Use as follows:
    - subclass
    - modify the behavior for what you want

    class TestModel(BaseTestModel):
        def training_step(...):
            # do your own thing

    or:

    model = BaseTestModel()
    model.validation_step = None

    """

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        """Forward pass of x through model."""
        return self.layer(x)

    @staticmethod
    def loss(_, prediction) -> torch.Tensor:
        """Arbitrary loss."""
        return torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))

    def step(self, x):
        """Single step in model."""
        x = self(x)
        return torch.nn.functional.mse_loss(x, torch.ones_like(x))

    def training_step(self, batch, batch_idx):
        """Single training step in model."""
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """Single validation step in the model."""
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"x": loss}

    def test_step(self, batch, batch_idx):
        """Single test step in the model."""
        output = self.layer(batch)
        loss = self.loss(batch, output)
        return {"y": loss}

    def configure_optimizers(self):
        """Configure which optimizer to use when training the model."""
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return {"optimizer": optimizer, "scheduler": lr_scheduler}

    def train_dataloader(self):
        """Define train dataloader used for training the model."""
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def val_dataloader(self):
        """Define validation dataloader used for validating the model."""
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

    def test_dataloader(self):
        """Define test dataloader used for testing the model."""
        return torch.utils.data.DataLoader(RandomDataset(32, 64))

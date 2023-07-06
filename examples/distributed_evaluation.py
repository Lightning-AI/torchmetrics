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
import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchmetrics
import torchvision.datasets as datasets
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics.utilities.distributed import EvaluationDistributedSampler


class DummyModel(torch.nn.Module):
    """Dummy model."""

    def __init__(self, n_feature: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_feature, 10)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.linear(x)


def use_lightning() -> None:
    """Use lightning to evaluate a model on a dataset."""
    from lightning.pytorch import LightningModule, Trainer


def _use_torch_worker(
    rank: int, model: torch.nn.Module, dataset: Dataset, batch_size: int, num_processes: int, gpu: bool
) -> None:
    """Worker function for torch.distributed evaluation."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nvcc" if gpu else "gloo", rank=rank, world_size=num_processes)

    device = torch.device(f"cuda:rank{rank}") if gpu else torch.device("cpu")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=EvaluationDistributedSampler(dataset, num_processes, rank),
    )

    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=10)
    metric = metric.to(device)

    batches, num_samples = 0, 0
    for _, batch in enumerate(dataloader):
        if gpu:
            batch = batch.cuda()
        preds = model(batch[0])
        target = batch[1]

        metric.update(preds.to(device), target.to(device))
        num_samples += len(target)
        batches += 1

    res = metric.compute()

    print(f"Rank {rank} processed {num_samples} samples and {batches} batches and calculated accuracy: {res}")


def use_torch(model: torch.nn.Module, dataset: Dataset, batch_size: int, num_processes: int, gpu: bool) -> None:
    """Use torch.distributed to evaluate a model on a dataset."""
    mp.spawn(_use_torch_worker, nprocs=2, args=(model, dataset, batch_size, num_processes, gpu))


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lightning", action="store_true")
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print(args)

    dataset = TensorDataset(torch.randn(199, 100), torch.randint(0, 10, (199,)))
    n_feature = 100
    dummy_model = DummyModel(n_feature)

    batch_size = 3
    if len(dataset) % (args.num_processes * batch_size) == 0:
        raise ValueError(
            "For this example the dataset size should NOT be divisible by the number of processes times the batch size."
        )

    if args.use_lightning:
        use_lightning(dummy_model, dataset, batch_size, args.num_processes, args.gpu)
    else:
        use_torch(dummy_model, dataset, batch_size, args.num_processes, args.gpu)


if __name__ == "__main__":
    main()

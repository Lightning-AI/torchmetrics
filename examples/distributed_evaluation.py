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
"""An example of how to use the distributed evaluation utilities in both Lightning and PyTorch.

To run using only Pytorch:
    python distributed_evaluation.py
To run using Lightning:
    python distributed_evaluation.py --use_lightning

By default, this example uses the EvaluationDistributedSampler, which is a custom sampler that ensures that no extra
samples are added to the dataset. This is important for evaluation, as we don't want to evaluate on the same samples
multiple times.

If you want to see the difference between the EvaluationDistributedSampler and the standard DistributedSampler, you
add the flag --use_standard. This will use the standard DistributedSampler, which will add extra samples to the dataset
and thus give incorrect results.

"""
import argparse
import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchmetrics
from lightning_utilities import module_available
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, DistributedSampler, TensorDataset
from torchmetrics.utilities.distributed import EvaluationDistributedSampler

_ = torch.manual_seed(42)


class DummyModel(Module):
    """Dummy model consisting of a single linear layer."""

    def __init__(self, n_feature: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_feature, 10)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.linear(x)


def calculate_accuracy_manually(dataset: Dataset, model: Module) -> Tensor:
    """Basic function to calculate accuracy manually, without any distributed stuff."""
    x, y = dataset.tensors
    preds = model(x)
    return (preds.argmax(dim=1) == y).float().mean()


def use_lightning(
    model: Module, dataset: Dataset, batch_size: int, use_standard: bool, num_processes: int, gpu: bool
) -> None:
    """Use lightning to evaluate a model on a dataset."""
    if module_available("lightning"):
        from lightning.pytorch import LightningModule, Trainer
    else:
        from pytorch_lightning import LightningModule, Trainer

    sampler_class = DistributedSampler if use_standard else EvaluationDistributedSampler

    class DummyLightningModule(LightningModule):
        def __init__(self, model: Module) -> None:
            super().__init__()
            self.model = model
            self.metric = torchmetrics.classification.MulticlassAccuracy(num_classes=10, average="micro")

        def forward(self, x: Tensor) -> Tensor:
            return self.model(x)

        def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
            preds = model(batch[0])
            target = batch[1]
            self.metric.update(preds, target)

        def on_test_epoch_end(self) -> None:
            self.log("test_acc", self.metric.compute())

        def test_dataloader(self) -> DataLoader:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler_class(dataset),
            )

    model = DummyLightningModule(model)

    trainer = Trainer(
        devices=num_processes,
        accelerator="cpu" if not gpu else "gpu",
    )

    res = trainer.test(model)
    manual_res = calculate_accuracy_manually(dataset, model)
    print(manual_res)
    if torch.allclose(torch.tensor(res[0]["test_acc"]), manual_res):
        print("success! result matched manual calculation")
    else:
        print("failure! result did not match manual calculation")


def _use_torch_worker_fn(
    rank: int, model: Module, dataset: Dataset, batch_size: int, use_standard: bool, num_processes: int, gpu: bool
) -> None:
    """Worker function for torch.distributed evaluation."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nvcc" if gpu else "gloo", rank=rank, world_size=num_processes)

    device = torch.device(f"cuda:rank{rank}") if gpu else torch.device("cpu")

    sampler_class = DistributedSampler if use_standard else EvaluationDistributedSampler

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler_class(dataset, num_processes, rank),
    )

    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=10, average="micro")
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

    manual_res = calculate_accuracy_manually(dataset, model)
    if torch.allclose(res, manual_res):
        print("success! result matched manual calculation")
    else:
        print("failure! result did not match manual calculation")


def use_torch(
    model: Module, dataset: Dataset, batch_size: int, use_standard: bool, num_processes: int, gpu: bool
) -> None:
    """Use torch.distributed to evaluate a model on a dataset."""
    mp.spawn(_use_torch_worker_fn, nprocs=2, args=(model, dataset, batch_size, use_standard, num_processes, gpu))


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lightning", action="store_true")
    parser.add_argument("--use_standard", action="store_true")
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
        use_lightning(dummy_model, dataset, batch_size, args.use_standard, args.num_processes, args.gpu)
    else:
        use_torch(dummy_model, dataset, batch_size, args.use_standard, args.num_processes, args.gpu)


if __name__ == "__main__":
    main()

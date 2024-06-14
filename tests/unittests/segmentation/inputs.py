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
__all__ = ["_Input"]

from typing import NamedTuple

from torch import Tensor

from unittests._helpers import seed_all

seed_all(42)


# extrinsic input for clustering metrics that requires predicted clustering labels and target clustering labels
class _Input(NamedTuple):
    preds: Tensor
    target: Tensor

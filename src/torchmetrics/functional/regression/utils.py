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
from torch import Tensor


def _check_data_shape_to_num_outputs(preds: Tensor, target: Tensor, num_outputs: int) -> None:
    """Check that predictions and target have the correct shape, else raise error."""
    if preds.ndim > 2 or target.ndim > 2:
        raise ValueError(
            f"Expected both predictions and target to be either 1- or 2-dimensional tensors,"
            f" but got {target.ndim} and {preds.ndim}."
        )
    cond1 = num_outputs == 1 and not (preds.ndim == 1 or preds.shape[1] == 1)
    cond2 = num_outputs > 1 and num_outputs != preds.shape[1]
    if cond1 or cond2:
        raise ValueError(
            f"Expected argument `num_outputs` to match the second dimension of input, but got {num_outputs}"
            f" and {preds.shape[1]}."
        )

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
from torchmetrics.utilities.checks import check_forward_full_state_property
from torchmetrics.utilities.data import apply_to_collection
from torchmetrics.utilities.distributed import class_reduce, reduce
from torchmetrics.utilities.prints import _future_warning, rank_zero_debug, rank_zero_info, rank_zero_warn

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

from torchmetrics.functional.nominal.cramers import cramers_v, cramers_v_matrix
from torchmetrics.functional.nominal.fleiss_kappa import fleiss_kappa
from torchmetrics.functional.nominal.pearson import (
    pearsons_contingency_coefficient,
    pearsons_contingency_coefficient_matrix,
)
from torchmetrics.functional.nominal.theils_u import theils_u, theils_u_matrix
from torchmetrics.functional.nominal.tschuprows import tschuprows_t, tschuprows_t_matrix
from torchmetrics.utilities.imports import _SCIPI_AVAILABLE

if _SCIPI_AVAILABLE:
    import scipy.signal

    # back compatibility patch due to SMRMpy using scipy.signal.hamming
    if not hasattr(scipy.signal, "hamming"):
        scipy.signal.hamming = scipy.signal.windows.hamming

__all__ = [
    "cramers_v",
    "cramers_v_matrix",
    "fleiss_kappa",
    "pearsons_contingency_coefficient",
    "pearsons_contingency_coefficient_matrix",
    "theils_u",
    "theils_u_matrix",
    "tschuprows_t",
    "tschuprows_t_matrix",
]

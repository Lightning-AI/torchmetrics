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
from torchmetrics.nominal.cramers import CramersV
from torchmetrics.nominal.fleiss_kappa import FleissKappa
from torchmetrics.nominal.pearson import PearsonsContingencyCoefficient
from torchmetrics.nominal.theils_u import TheilsU
from torchmetrics.nominal.tschuprows import TschuprowsT

__all__ = [
    "CramersV",
    "FleissKappa",
    "PearsonsContingencyCoefficient",
    "TheilsU",
    "TschuprowsT",
]

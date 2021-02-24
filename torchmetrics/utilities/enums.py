# Copyright The PyTorch Lightning team.
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
from enum import Enum
from typing import Union


class EnumStr(str, Enum):
    """ Type of any enumerator with allowed comparison to string invariant to cases. """

    @classmethod
    def from_str(cls, value: str) -> 'EnumStr':
        statuses = [status for status in dir(cls) if not status.startswith('_')]
        for st in statuses:
            if st.lower() == value.lower():
                return getattr(cls, st)
        return None

    def __eq__(self, other: Union[str, Enum]) -> bool:
        other = other.value if isinstance(other, Enum) else str(other)
        return self.value.lower() == other.lower()

    def __hash__(self):
        # re-enable hashtable so it can be used as a dict key or in a set
        # example: set(LightningEnum)
        return hash(self.name)


class DataType(EnumStr):
    """Enum to represent data type"""

    BINARY = "binary"
    MULTILABEL = "multi-label"
    MULTICLASS = "multi-class"
    MULTIDIM_MULTICLASS = "multi-dim multi-class"


class AverageMethod(EnumStr):
    """Enum to represent average method"""

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    NONE = "none"
    SAMPLES = "samples"


class SimpleAverageMethod(EnumStr):
    """Enum to represent average method fro AUC"""
    MACRO = 'macro'
    WEIGHTED = 'weighted'
    NONE = None


class MDMCAverageMethod(EnumStr):
    """Enum to represent multi-dim multi-class average method"""

    GLOBAL = "global"
    SAMPLEWISE = "samplewise"

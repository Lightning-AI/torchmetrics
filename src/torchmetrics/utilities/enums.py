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
from typing import Optional

from lightning_utilities.core.enums import StrEnum as StrEnum


class EnumStr(StrEnum):
    @property
    def _name(self) -> str:
        return "Task"

    @classmethod
    def from_str(cls, value: str) -> Optional["EnumStr"]:
        """Raises:
        ValueError:
        If required information measure is not among the supported options.
        """
        _allowed_im = [im.lower() for im in cls._member_names_]

        enum_key = super().from_str(value)
        if enum_key is not None and enum_key in _allowed_im:
            return enum_key
        raise ValueError(f"Invalid {cls._name}: expected one of {_allowed_im}, but got {enum_key}.")


class DataType(EnumStr):
    """Enum to represent data type.

    >>> "Binary" in list(DataType)
    True
    """

    @property
    def _name(self) -> str:
        return "Data type"

    BINARY = "binary"
    MULTILABEL = "multi-label"
    MULTICLASS = "multi-class"
    MULTIDIM_MULTICLASS = "multi-dim multi-class"


class AverageMethod(EnumStr):
    """Enum to represent average method.

    >>> None in list(AverageMethod)
    True
    >>> AverageMethod.NONE == None
    True
    >>> AverageMethod.NONE == 'none'
    True
    """

    @property
    def _name(self) -> str:
        return "Average method"

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    NONE = None
    SAMPLES = "samples"


class MDMCAverageMethod(EnumStr):
    """Enum to represent multi-dim multi-class average method."""

    @property
    def _name(self) -> str:
        return "MDMC Average method"

    GLOBAL = "global"
    SAMPLEWISE = "samplewise"


class ClassificationTask(EnumStr):
    """Enum to represent the different tasks in classification metrics.

    >>> "binary" in list(ClassificationTask)
    True
    """

    @property
    def _name(self) -> str:
        return "Classification"

    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"

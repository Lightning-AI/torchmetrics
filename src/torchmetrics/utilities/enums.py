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
from typing import Union

from lightning_utilities.core.enums import StrEnum
from typing_extensions import Literal


class EnumStr(StrEnum):
    """Base Enum."""

    @staticmethod
    def _name() -> str:
        return "Task"

    @classmethod
    def from_str(cls, value: str, source: Literal["key", "value", "any"] = "key") -> "EnumStr":
        """Load from string.

        Raises:
            ValueError:
                If required value is not among the supported options.

        >>> class MyEnum(EnumStr):
        ...     a = "aaa"
        ...     b = "bbb"
        >>> MyEnum.from_str("a")
        <MyEnum.a: 'aaa'>
        >>> MyEnum.from_str("c")
        Traceback (most recent call last):
          ...
        ValueError: Invalid Task: expected one of ['a', 'b'], but got c.
        """
        try:
            me = super().from_str(value.replace("-", "_"), source=source)
        except ValueError as err:
            _allowed_im = [m.lower() for m in cls._member_names_]
            raise ValueError(
                f"Invalid {cls._name()}: expected one of {cls._allowed_matches(source)}, but got {value}."
            ) from err
        return cls(me)


class DataType(EnumStr):
    """Enum to represent data type.

    >>> "Binary" in list(DataType)
    True
    """

    @staticmethod
    def _name() -> str:
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

    @staticmethod
    def _name() -> str:
        return "Average method"

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    NONE = None
    SAMPLES = "samples"


class MDMCAverageMethod(EnumStr):
    """Enum to represent multi-dim multi-class average method."""

    @staticmethod
    def _name() -> str:
        return "MDMC Average method"

    GLOBAL = "global"
    SAMPLEWISE = "samplewise"


class ClassificationTask(EnumStr):
    """Enum to represent the different tasks in classification metrics.

    >>> "binary" in list(ClassificationTask)
    True
    """

    @staticmethod
    def _name() -> str:
        return "Classification"

    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class ClassificationTaskNoBinary(EnumStr):
    """Enum to represent the different tasks in classification metrics.

    >>> "binary" in list(ClassificationTaskNoBinary)
    False
    """

    @staticmethod
    def _name() -> str:
        return "Classification"

    MULTILABEL = "multilabel"
    MULTICLASS = "multiclass"


class ClassificationTaskNoMultilabel(EnumStr):
    """Enum to represent the different tasks in classification metrics.

    >>> "multilabel" in list(ClassificationTaskNoMultilabel)
    False
    """

    @staticmethod
    def _name() -> str:
        return "Classification"

    BINARY = "binary"
    MULTICLASS = "multiclass"

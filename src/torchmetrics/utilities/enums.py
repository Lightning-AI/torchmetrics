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

from lightning_utilities.core.enums import StrEnum


class DataType(StrEnum):
    """Enum to represent data type.

    >>> "Binary" in list(DataType)
    True
    """

    BINARY = "binary"
    MULTILABEL = "multi-label"
    MULTICLASS = "multi-class"
    MULTIDIM_MULTICLASS = "multi-dim multi-class"


class AverageMethod(StrEnum):
    """Enum to represent average method.

    >>> None in list(AverageMethod)
    True
    >>> AverageMethod.NONE == None
    True
    >>> AverageMethod.NONE == 'none'
    True
    """

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    NONE = None
    SAMPLES = "samples"


class MDMCAverageMethod(StrEnum):
    """Enum to represent multi-dim multi-class average method."""

    GLOBAL = "global"
    SAMPLEWISE = "samplewise"


class ClassificationTask(StrEnum):
    """Enum to represent the different tasks in classification metrics.

    >>> "binary" in list(ClassificationTask)
    True
    """

    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"

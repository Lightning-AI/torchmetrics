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
"""Import utilities."""
import operator
from collections import OrderedDict  # noqa: F401
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Callable, Optional

from packaging.version import Version
from pkg_resources import DistributionNotFound, get_distribution


@lru_cache()
def _package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    >>> _package_available('os')
    True
    >>> _package_available('bla')
    False
    """
    try:
        return find_spec(package_name) is not None
    except AttributeError:
        # Python 3.6
        return False
    except (ImportError, ModuleNotFoundError):
        # Python 3.7+
        return False


@lru_cache()
def _module_available(module_path: str) -> bool:
    """Check if a module path is available in your environment.

    >>> _module_available('os')
    True
    >>> _module_available('os.bla')
    False
    >>> _module_available('bla.bla')
    False
    """
    module_names = module_path.split(".")
    if not _package_available(module_names[0]):
        return False
    try:
        module = import_module(module_names[0])
    except ImportError:
        return False
    for name in module_names[1:]:
        if not hasattr(module, name):
            return False
        module = getattr(module, name)
    return True


@lru_cache()
def _compare_version(package: str, op: Callable, version: str) -> Optional[bool]:
    """Compare package version with some requirements.

    >>> import operator
    >>> _compare_version("torch", operator.ge, "0.1")
    True
    >>> _compare_version("any_module", operator.ge, "0.0")  # is None
    """
    if not _module_available(package):
        return None
    try:
        pkg = import_module(package)
        pkg_version = pkg.__version__  # type: ignore
    except (ModuleNotFoundError, DistributionNotFound):
        return None
    except (ImportError, AttributeError):
        # catches cyclic imports - the case with integrated libs
        # see: https://stackoverflow.com/a/32965521
        pkg_version = get_distribution(package).version
    try:
        pkg_version = Version(pkg_version)
    except TypeError:
        # this is mock by sphinx, so it shall return True ro generate all summaries
        return True
    return op(pkg_version, Version(version))


_TORCH_LOWER_1_4: Optional[bool] = _compare_version("torch", operator.lt, "1.4.0")
_TORCH_LOWER_1_5: Optional[bool] = _compare_version("torch", operator.lt, "1.5.0")
_TORCH_LOWER_1_6: Optional[bool] = _compare_version("torch", operator.lt, "1.6.0")
_TORCH_LOWER_1_12_DEV: Optional[bool] = _compare_version("torch", operator.lt, "1.12.0.dev")
_TORCH_GREATER_EQUAL_1_6: Optional[bool] = _compare_version("torch", operator.ge, "1.6.0")
_TORCH_GREATER_EQUAL_1_7: Optional[bool] = _compare_version("torch", operator.ge, "1.7.0")
_TORCH_GREATER_EQUAL_1_8: Optional[bool] = _compare_version("torch", operator.ge, "1.8.0")
_TORCH_GREATER_EQUAL_1_10: Optional[bool] = _compare_version("torch", operator.ge, "1.10.0")
_TORCH_GREATER_EQUAL_1_11: Optional[bool] = _compare_version("torch", operator.ge, "1.11.0")

_JIWER_AVAILABLE: bool = _package_available("jiwer")
_NLTK_AVAILABLE: bool = _package_available("nltk")
_ROUGE_SCORE_AVAILABLE: bool = _package_available("rouge_score")
_BERTSCORE_AVAILABLE: bool = _package_available("bert_score")
_SCIPY_AVAILABLE: bool = _package_available("scipy")
_TORCH_FIDELITY_AVAILABLE: bool = _package_available("torch_fidelity")
_LPIPS_AVAILABLE: bool = _package_available("lpips")
_PYCOCOTOOLS_AVAILABLE: bool = _package_available("pycocotools")
_TORCHVISION_AVAILABLE: bool = _package_available("torchvision")
_TORCHVISION_GREATER_EQUAL_0_8: Optional[bool] = _compare_version("torchvision", operator.ge, "0.8.0")
_TQDM_AVAILABLE: bool = _package_available("tqdm")
_TRANSFORMERS_AVAILABLE: bool = _package_available("transformers")
_TRANSFORMERS_AUTO_AVAILABLE = _module_available("transformers.models.auto")
_PESQ_AVAILABLE: bool = _package_available("pesq")
_SACREBLEU_AVAILABLE: bool = _package_available("sacrebleu")
_REGEX_AVAILABLE: bool = _package_available("regex")
_PYSTOI_AVAILABLE: bool = _package_available("pystoi")
_FAST_BSS_EVAL_AVAILABLE: bool = _package_available("fast_bss_eval")

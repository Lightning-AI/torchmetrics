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
"""Import utilities."""
import operator
import shutil
import sys
from distutils.version import LooseVersion
from typing import Optional

from lightning_utilities.core.imports import compare_version, package_available

_PYTHON_VERSION = ".".join(map(str, [sys.version_info.major, sys.version_info.minor, sys.version_info.micro]))
_PYTHON_LOWER_3_8 = LooseVersion(_PYTHON_VERSION) < LooseVersion("3.8")
_TORCH_LOWER_1_12_DEV: Optional[bool] = compare_version("torch", operator.lt, "1.12.0.dev")
_TORCH_GREATER_EQUAL_1_9: Optional[bool] = compare_version("torch", operator.ge, "1.9.0")
_TORCH_GREATER_EQUAL_1_10: Optional[bool] = compare_version("torch", operator.ge, "1.10.0")
_TORCH_GREATER_EQUAL_1_11: Optional[bool] = compare_version("torch", operator.ge, "1.11.0")
_TORCH_GREATER_EQUAL_1_12: Optional[bool] = compare_version("torch", operator.ge, "1.12.0")
_TORCH_GREATER_EQUAL_1_13: Optional[bool] = compare_version("torch", operator.ge, "1.13.0")

_JIWER_AVAILABLE: bool = package_available("jiwer")
_NLTK_AVAILABLE: bool = package_available("nltk")
_ROUGE_SCORE_AVAILABLE: bool = package_available("rouge_score")
_BERTSCORE_AVAILABLE: bool = package_available("bert_score")
_SCIPY_AVAILABLE: bool = package_available("scipy")
_TORCH_FIDELITY_AVAILABLE: bool = package_available("torch_fidelity")
_LPIPS_AVAILABLE: bool = package_available("lpips")
_PYCOCOTOOLS_AVAILABLE: bool = package_available("pycocotools")
_TORCHVISION_AVAILABLE: bool = package_available("torchvision")
_TORCHVISION_GREATER_EQUAL_0_8: Optional[bool] = compare_version("torchvision", operator.ge, "0.8.0")
_TORCHVISION_GREATER_EQUAL_0_13: Optional[bool] = compare_version("torchvision", operator.ge, "0.13.0")
_TQDM_AVAILABLE: bool = package_available("tqdm")
_TRANSFORMERS_AVAILABLE: bool = package_available("transformers")
_PESQ_AVAILABLE: bool = package_available("pesq")
_GAMMATONE_AVAILABEL: bool = package_available("gammatone")
_TORCHAUDIO_AVAILABEL: bool = package_available("torchaudio")
_TORCHAUDIO_GREATER_EQUAL_0_10: Optional[bool] = compare_version("torchaudio", operator.ge, "0.10.0")
_SACREBLEU_AVAILABLE: bool = package_available("sacrebleu")
_REGEX_AVAILABLE: bool = package_available("regex")
_PYSTOI_AVAILABLE: bool = package_available("pystoi")
_FAST_BSS_EVAL_AVAILABLE: bool = package_available("fast_bss_eval")
_MATPLOTLIB_AVAILABLE: bool = package_available("matplotlib")
_SCIENCEPLOT_AVAILABLE: bool = package_available("scienceplots")
_MULTIPROCESSING_AVAILABLE: bool = package_available("multiprocessing")
_XLA_AVAILABLE: bool = package_available("torch_xla")

_LATEX_AVAILABLE: bool = shutil.which("latex") is not None

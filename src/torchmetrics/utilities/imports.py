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
from typing import Optional

from lightning_utilities.core.imports import compare_version, package_available
from packaging.version import Version, parse

_PYTHON_VERSION = ".".join(map(str, [sys.version_info.major, sys.version_info.minor, sys.version_info.micro]))
_PYTHON_LOWER_3_8 = parse(_PYTHON_VERSION) < Version("3.8")
_TORCH_LOWER_2_0: Optional[bool] = compare_version("torch", operator.lt, "2.0.0")
_TORCH_LOWER_1_12_DEV: Optional[bool] = compare_version("torch", operator.lt, "1.12.0.dev")
_TORCH_GREATER_EQUAL_1_9: Optional[bool] = compare_version("torch", operator.ge, "1.9.0")
_TORCH_GREATER_EQUAL_1_10: Optional[bool] = compare_version("torch", operator.ge, "1.10.0")
_TORCH_GREATER_EQUAL_1_11: Optional[bool] = compare_version("torch", operator.ge, "1.11.0")
_TORCH_GREATER_EQUAL_1_12: Optional[bool] = compare_version("torch", operator.ge, "1.12.0")
_TORCH_GREATER_EQUAL_1_13: Optional[bool] = compare_version("torch", operator.ge, "1.13.0")
_TORCH_GREATER_EQUAL_2_0: Optional[bool] = compare_version("torch", operator.ge, "2.0.0")
_TORCH_GREATER_EQUAL_2_1: Optional[bool] = compare_version("torch", operator.ge, "2.1.0")

_JIWER_AVAILABLE: bool = package_available("jiwer")
_NLTK_AVAILABLE: bool = package_available("nltk")
_ROUGE_SCORE_AVAILABLE: bool = package_available("rouge_score")
_BERTSCORE_AVAILABLE: bool = package_available("bert_score")
_SCIPY_AVAILABLE: bool = package_available("scipy")
_SCIPY_GREATER_EQUAL_1_8 = compare_version("scipy", operator.ge, "1.8.0")
_TORCH_FIDELITY_AVAILABLE: bool = package_available("torch_fidelity")
_LPIPS_AVAILABLE: bool = package_available("lpips")
_PYCOCOTOOLS_AVAILABLE: bool = package_available("pycocotools")
_TORCHVISION_AVAILABLE: bool = package_available("torchvision")
_TORCHVISION_GREATER_EQUAL_0_8: Optional[bool] = compare_version("torchvision", operator.ge, "0.8.0")
_TORCHVISION_GREATER_EQUAL_0_13: Optional[bool] = compare_version("torchvision", operator.ge, "0.13.0")
_TQDM_AVAILABLE: bool = package_available("tqdm")
_TRANSFORMERS_AVAILABLE: bool = package_available("transformers")
_TRANSFORMERS_GREATER_EQUAL_4_4: Optional[bool] = compare_version("transformers", operator.ge, "4.4.0")
_TRANSFORMERS_GREATER_EQUAL_4_10: Optional[bool] = compare_version("transformers", operator.ge, "4.10.0")
_PESQ_AVAILABLE: bool = package_available("pesq")
_GAMMATONE_AVAILABLE: bool = package_available("gammatone")
_TORCHAUDIO_AVAILABLE: bool = package_available("torchaudio")
_TORCHAUDIO_GREATER_EQUAL_0_10: Optional[bool] = compare_version("torchaudio", operator.ge, "0.10.0")
_SACREBLEU_AVAILABLE: bool = package_available("sacrebleu")
_REGEX_AVAILABLE: bool = package_available("regex")
_PYSTOI_AVAILABLE: bool = package_available("pystoi")
_FAST_BSS_EVAL_AVAILABLE: bool = package_available("fast_bss_eval")
_MATPLOTLIB_AVAILABLE: bool = package_available("matplotlib")
_SCIENCEPLOT_AVAILABLE: bool = package_available("scienceplots")
_MULTIPROCESSING_AVAILABLE: bool = package_available("multiprocessing")
_XLA_AVAILABLE: bool = package_available("torch_xla")
_PIQ_GREATER_EQUAL_0_8: Optional[bool] = compare_version("piq", operator.ge, "0.8.0")
_FASTER_COCO_EVAL_AVAILABLE: bool = package_available("faster_coco_eval")

_LATEX_AVAILABLE: bool = shutil.which("latex") is not None

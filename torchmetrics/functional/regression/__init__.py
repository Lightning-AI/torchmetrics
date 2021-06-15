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
from torchmetrics.functional.regression.explained_variance import explained_variance  # noqa: F401
from torchmetrics.functional.regression.mean_absolute_error import mean_absolute_error  # noqa: F401
from torchmetrics.functional.regression.mean_absolute_percentage_error import (  # noqa: F401
    mean_absolute_percentage_error,
)
from torchmetrics.functional.regression.mean_squared_error import mean_squared_error  # noqa: F401
from torchmetrics.functional.regression.mean_squared_log_error import mean_squared_log_error  # noqa: F401
from torchmetrics.functional.regression.pearson import pearson_corrcoef  # noqa: F401
from torchmetrics.functional.regression.psnr import psnr  # noqa: F401
from torchmetrics.functional.regression.r2score import r2score  # noqa: F401
from torchmetrics.functional.regression.spearman import spearman_corrcoef  # noqa: F401
from torchmetrics.functional.regression.ssim import ssim  # noqa: F401

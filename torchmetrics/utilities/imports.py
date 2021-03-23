from distutils.version import LooseVersion

import torch

_TORCH_LOWER_1_4 = LooseVersion(torch.__version__) < LooseVersion("1.4.0")
_TORCH_LOWER_1_5 = LooseVersion(torch.__version__) < LooseVersion("1.5.0")
_TORCH_LOWER_1_6 = LooseVersion(torch.__version__) < LooseVersion("1.6.0")

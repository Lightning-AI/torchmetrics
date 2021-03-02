from distutils.version import LooseVersion

import torch

_MARK_TORCH_MIN_1_4 = dict(condition=LooseVersion(torch.__version__) < LooseVersion("1.4"), reason='required PT >= 1.4')
_MARK_TORCH_MIN_1_5 = dict(condition=LooseVersion(torch.__version__) < LooseVersion("1.5"), reason='required PT >= 1.5')
_MARK_TORCH_MIN_1_6 = dict(condition=LooseVersion(torch.__version__) < LooseVersion("1.6"), reason='required PT >= 1.6')

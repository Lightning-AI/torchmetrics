from typing import List, Optional, Sequence, Tuple, Union

import torch
from deprecate import deprecated, void
from torch import Tensor
from torch.nn import functional as F
from typing_extensions import Literal

from torchmetrics.utilities import _future_warning
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce


def filter_process(img, fltr, mode="same"):
    output = F.conv2d(img.unsqueeze(0).unsqueeze(0), torch.rot90(fltr, 2).unsqueeze(0).unsqueeze(0), padding=mode)

    return output.squeeze(0).squeeze(0)


def _get_sums(Input_Image, Noisy_Image, win, mode="same"):
    mu1, mu2 = (filter_process(Input_Image, win, mode), filter_process(Noisy_Image, win, mode))
    return mu1 * mu1, mu2 * mu2, mu1 * mu2


def _get_sigmas(Input_Image, Noisy_Image, win, mode="same"):
    Input_Image_sum_sq, Noisy_Image_sum_sq, Input_Image_Noisy_Image_sum_mul = _get_sums(
        Input_Image, Noisy_Image, win, mode
    )
    outputs = (
        filter_process(Input_Image * Input_Image, win, mode) - Input_Image_sum_sq,
        filter_process(Noisy_Image * Noisy_Image, win, mode) - Noisy_Image_sum_sq,
        filter_process(Input_Image * Noisy_Image, win, mode) - Input_Image_Noisy_Image_sum_mul,
    )

    return outputs


def _scc_single(Input_Image, Noisy_Image, Window, Window_Size):

    win = torch.ones((Window_Size, Window_Size)) / Window_Size ** 2
    sigma_Input_Image_sq, sigma_Noisy_Image_sq, sigma_Input_Image_Noisy_Image = _get_sigmas(
        Input_Image, Noisy_Image, win
    )
    sigma_Input_Image_sq[sigma_Input_Image_sq < 0] = 0
    sigma_Noisy_Image_sq[sigma_Noisy_Image_sq < 0] = 0

    den = torch.sqrt(sigma_Input_Image_sq) * torch.sqrt(sigma_Noisy_Image_sq)

    idx = den == 0

    den[den == 0] = 1

    scc = sigma_Input_Image_Noisy_Image / den

    scc[idx] = 0

    return scc


def spatial_correlation_coefficient(
    Input_Image: Tensor,
    Noisy_Image: Tensor,
    Window: List[int] = [[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]],
    Window_Size: int = 8,
):
    """calculates spatial correlation coeffcient.

    :param Input_Image: first original image
    :param Noisy_Image: second deformed image
    :param Window: High Pass Filter for spatial processing
    :param Window_Size: sliding window size default = 8
    :returns:  float -- scc value.
    """

    coefs = torch.zeros(Input_Image.shape)

    for i in range(Input_Image.shape[0]):
        coefs[i, :, :] = _scc_single(Input_Image[i, :, :], Noisy_Image[i, :, :], Window, Window_Size)

    scc = torch.mean(coefs)

    return scc


if __name__ == "__main__":
    a = torch.rand((3, 10, 10))
    b = torch.rand((3, 10, 10))
    c = torch.rand((3, 3))
    ots = spatial_correlation_coefficient(a, b, c, 8)
    print(ots)

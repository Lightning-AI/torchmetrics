import torch


def total_variation(img: torch.Tensor) -> torch.Tensor:
    """Computes total variation loss.

    Adapted from https://github.com/jxgu1016/Total_Variation_Loss.pytorch
    Args:
        img (torch.Tensor): A NCHW image batch.

    Returns:
        A loss scalar value.
    """

    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    _height = img.size()[2]
    _width = img.size()[3]
    _count_height = tensor_size(img[:, :, 1:, :])
    _count_width = tensor_size(img[:, :, :, 1:])
    _height_tv = torch.pow((img[:, :, 1:, :] - img[:, :, : _height - 1, :]), 2).sum()
    _width_tv = torch.pow((img[:, :, :, 1:] - img[:, :, :, : _width - 1]), 2).sum()
    return (2 * (_height_tv / _count_height + _width_tv / _count_width)) / img.size()[0]

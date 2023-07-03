import torch
from torch import Tensor, tensor
from torch.nn.functional import conv2d


def _filter(win_size: float, sigma: float) -> Tensor:
    # This code is inspired by
    # https://github.com/andrewekhalel/sewar/blob/ac76e7bc75732fde40bb0d3908f4b6863400cc27/sewar/utils.py#L45
    start: float = -win_size // 2 + 1
    end: float = win_size // 2 + 1
    x, y = torch.meshgrid(torch.arange(start, end), torch.arange(start, end), indexing="ij")
    g = torch.exp(-torch.div(x ** 2 + y ** 2, 2.0 * tensor(sigma) ** 2))
    g[g < torch.finfo(g.dtype).eps * g.max()] = 0

    if not g.size() == (win_size, win_size):
        raise ValueError(f"Resulting filter size of {g.size()} does not match expected size {(win_size, win_size)}!")

    den = torch.sum(g)
    if den != 0:
        g = torch.div(g, den)
    return g


def visual_information_fidelity(preds: Tensor, target: Tensor, sigma_n_sq: float = 2.0) -> Tensor:
    """Compute Pixel Based Visual Information Fidelity (vif-p).

    Args:
        preds: predicted images
        target: ground truth images
        sigma_n_sq: variance of the visual noise

    Return:
        Tensor with vif-p score

    Raises:
        ValueError:
            If ``data_range`` is neither a ``tuple`` nor a ``float``
    """
    # This code is inspired by
    # https://github.com/photosynthesis-team/piq/blob/01e16b7d8c76bc8765fb6a69560d806148b8046a/piq/vif.py and
    # https://github.com/andrewekhalel/sewar/blob/ac76e7bc75732fde40bb0d3908f4b6863400cc27/sewar/full_ref.py#L357.
    #
    # Reference: https://ieeexplore.ieee.org/abstract/document/1576816
    # https://live.ece.utexas.edu/research/quality/VIF.htm

    # Constant for numerical stability
    eps = torch.tensor(1e-10)

    sigma_n_sq = torch.tensor(sigma_n_sq)

    preds_vif, target_vif = torch.zeros(1), torch.zeros(1)
    for scale in range(4):
        n = 2.0 ** (4 - scale) + 1
        kernel = _filter(n, n / 5)[None, None, :]

        if scale > 0:
            target = conv2d(target, kernel)[:, :, ::2, ::2]
            preds = conv2d(preds, kernel)[:, :, ::2, ::2]

        mu_target = conv2d(target, kernel)
        mu_preds = conv2d(preds, kernel)
        mu_target_sq = mu_target ** 2
        mu_preds_sq = mu_preds ** 2
        mu_target_preds = mu_target * mu_preds

        sigma_target_sq = torch.clamp(conv2d(target ** 2, kernel) - mu_target_sq, min=0.0)
        sigma_preds_sq = torch.clamp(conv2d(preds ** 2, kernel) - mu_preds_sq, min=0.0)
        sigma_target_preds = conv2d(target * preds, kernel) - mu_target_preds

        g = sigma_target_preds / (sigma_target_sq + eps)
        sigma_v_sq = sigma_preds_sq - g * sigma_target_preds

        mask = sigma_target_sq < eps
        g[mask] = 0
        sigma_v_sq[mask] = sigma_preds_sq[mask]
        sigma_target_sq[mask] = 0

        mask = sigma_preds_sq < eps
        g[mask] = 0
        sigma_v_sq[mask] = 0

        mask = g < 0
        sigma_v_sq[mask] = sigma_preds_sq[mask]
        g[mask] = 0
        sigma_v_sq = torch.clamp(sigma_v_sq, min=eps)

        preds_vif_scale = torch.log10(1.0 + (g ** 2.0) * sigma_target_sq / (sigma_v_sq + sigma_n_sq))
        preds_vif = preds_vif + torch.sum(preds_vif_scale, dim=[1, 2, 3])
        target_vif = target_vif + torch.sum(torch.log10(1.0 + sigma_target_sq / sigma_n_sq), dim=[1, 2, 3])
    return preds_vif / target_vif

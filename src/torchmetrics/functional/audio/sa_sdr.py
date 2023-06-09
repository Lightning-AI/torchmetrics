

def source_aggregated_signal_distortion_ratio(preds: Tensor, target: Tensor, zero_mean: bool = False) -> Tensor:
    """`Scale-invariant signal-to-distortion ratio`_ (SI-SDR).

    The SI-SDR value is in general considered an overall measure of how good a source sound.

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``
        zero_mean: If to zero mean target and preds or not

    Returns:
        Float tensor with shape ``(...,)`` of SDR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape

    Example:
        >>> from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> scale_invariant_signal_distortion_ratio(preds, target)
        tensor(18.4030)
    """
    _check_same_shape(preds, target)
    eps = torch.finfo(preds.dtype).eps

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) / (
        torch.sum(target**2, dim=-1, keepdim=True) + eps
    )
    target_scaled = alpha * target

    noise = target_scaled - preds

    val = (torch.sum(target_scaled**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    return 10 * torch.log10(val)

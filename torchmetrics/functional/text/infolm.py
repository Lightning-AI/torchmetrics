import os
from enum import unique
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.utilities.enums import EnumStr
from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE
from torchmetrics.functional.text.helper_embedding_metric import TextDataset

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
else:
    __doctest_skip__ = ["infolm"]


_ALLOWED_INFORMATION_MEASURE = (
    "kl_divergence",
    "alpha_divergence",
    "beta_divergence",
    "ab_divergence"
    "renyi_divergence",
    "l1_distance",
    "l2_distance",
    "l_infinity_distance",
    "fisher_rao_distance",
)


_ALLOWED_INFORMATION_MEASURE_LITERAL = Literal[
    "kl_divergence",
    "alpha_divergence",
    "beta_divergence",
    "ab_divergence"
    "renyi_divergence",
    "l1_distance",
    "l2_distance",
    "l_infinity_distance",
    "fisher_rao_distance",
]


@unique
class _IMEnum(EnumStr):
    """
    """

    KL_DIVERGENCE = "kl_divergence"
    ALPHA_DIVERGENCE = "alpha_divergence"
    BETA_DIVERGENCE = "beta_divergence"
    AB_DIVERGENCE = "ab_divergence"
    RENYI_DIVERGENCE = "renyi_divergence"
    L1_DISTANCE = "l1_distance"
    L2_DISTANCE = "l2_distance"
    L_INFINITY_DISTANCE = "l_infinity_distance"
    FISHER_RAO_DISTANCE = "fisher_rao_distance"

    @classmethod
    def from_str(cls, value: str) -> Optional["EnumStr"]:
        statuses = [status for status in dir(cls) if not status.startswith("_")]
        for st in statuses:
            if st.lower() == value.lower():
                return getattr(cls, st)
        raise ValueError(f"Invalid information measure got. Please use one of {_ALLOWED_INFORMATION_MEASURE}.")


class _InformationMeasure:
    def __init__(
        self,
        information_measure: _ALLOWED_INFORMATION_MEASURE_LITERAL,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> None:
        self.information_measure = _IMEnum.from_str(information_measure)
        if self.information_measure in [_IMEnum.ALPHA_DIVERGENCE, _IMEnum.AB_DIVERGENCE, _IMEnum.RENYI_DIVERGENCE]:
            if not isinstance(alpha, float):
                raise ValueError(f"Parameter `alpha` is expected to be defined for {information_measure}.")
        if self.information_measure in [_IMEnum.BETA_DIVERGENCE, _IMEnum.AB_DIVERGENCE]:
            if not isinstance(beta, float):
                raise ValueError(f"Parameter `beta` is expected to be defined for {information_measure}.")
        if self.information_measure == _IMEnum.ALPHA_DIVERGENCE and alpha in [0, 1]:
            raise ValueError(f"Parameter `alpha` is expected to be differened from 0 and 1 for {information_measure}.")
        if self.information_measure == _IMEnum.BETA_DIVERGENCE and alpha in [0, -1]:
            raise ValueError(f"Parameter `beta` is expected to be differened from 0 and -1 for {information_measure}.")
        if self.information_measure == _IMEnum.AB_DIVERGENCE and 0 in [alpha, beta, alpha + beta]:
            raise ValueError(
                "Parameters `alpha`, `beta` and their sum are expected to be differened from 0 for "
                f"{information_measure}."
            )

        self.alpha = alpha
        self.beta = beta

    def __call__(self, preds_distribution: Tensor, target_distribtuion: Tensor) -> Tensor:
        information_measure_function = getattr(self, f"_calculate_{self.information_measure}")
        return information_measure_function(preds_distribution, target_distribtuion)

    @staticmethod
    def _calculate_kl_divergence(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """
        Args:
            preds_distribution:
            target_distribution
        """
        return torch.sum(preds_distribution * torch.log(preds_distribution / target_distribution), dim=-1)

    def _calculate_alpha_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        _alpha_denom = self.alpha * (self.alpha - 1)
        alpha_divergence = (
            1 - torch.sum(target_distribution ** self.alpha * preds_distribution ** (1 - self.alpha), dim=-1)
        ) / _alpha_denom
        return alpha_divergence

    def _calculate_ab_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        x = torch.log(torch.sum(target_distribution ** (self.beta + self.alpha), dim=-1))
        y = torch.log(torch.sum(preds_distribution ** (self.beta + self.alpha), dim=-1))
        z = torch.log(torch.sum(target_distribution ** self.alpha * preds_distribution ** self.beta, dim=-1))
        ab_divergence = (
            x / (self.beta * (self.beta + self.alpha)) + y / (self.beta + self.alpha) - z / (self.alpha * self.beta)
        )
        return ab_divergence

    def _calculate_beta_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        self.alpha = 1.0
        beta_divergence = self._calculate_ab_divergence(preds_distribution, target_distribution)
        return beta_divergence


def _load_tokenizer_and_model(
    model_name_or_path: Union[str, os.PathLike], device: Union[str, torch.device]
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Args:
        model_name_or_path:
        device:

    Return:
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, output_hidden_states=True)
    model.eval()
    model.to(device)
    return tokenizer, model


def _infolm_update():
    pass


def _infolm_compute():
    pass


def infolm(
    preds: Union[str, Sequence[str]],
    target: Sequence[Union[str, Sequence[str]]],
    model_name_or_path: Union[str, os.PathLike] = "bert-base-uncased",
    temperature: float = 0.25,
    information_measure: _ALLOWED_INFORMATION_MEASURE_LITERAL = "kl_divergence",
    idf: bool = True,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Args:
        preds:
        target:
        model_name_or_path:
        temperature:
        information_measure:
        idf:
        alpha:
        beta:
        device:
    """
    tokenizer, model = _load_tokenizer_and_model(model_name_or_path, device)

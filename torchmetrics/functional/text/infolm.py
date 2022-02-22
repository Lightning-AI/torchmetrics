import os
from enum import unique
from typing import Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.functional.text.helper_embedding_metric import TextDataset
from torchmetrics.utilities.enums import EnumStr
from torchmetrics.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
else:
    __doctest_skip__ = ["infolm"]


_ALLOWED_INFORMATION_MEASURE = (
    "kl_divergence",
    "alpha_divergence",
    "beta_divergence",
    "ab_divergence" "renyi_divergence",
    "l1_distance",
    "l2_distance",
    "l_infinity_distance",
    "fisher_rao_distance",
)


_ALLOWED_INFORMATION_MEASURE_LITERAL = Literal[
    "kl_divergence",
    "alpha_divergence",
    "beta_divergence",
    "ab_divergence" "renyi_divergence",
    "l1_distance",
    "l2_distance",
    "l_infinity_distance",
    "fisher_rao_distance",
]


@unique
class _IMEnum(EnumStr):
    """A helper Enum class for storing the information measure."""

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
        """
        Raises:
            ValueError:
                If required information measure is not among the supported options.
        """
        statuses = [status for status in dir(cls) if not status.startswith("_")]
        for st in statuses:
            if st.lower() == value.lower():
                return getattr(cls, st)
        raise ValueError(f"Invalid information measure got. Please use one of {_ALLOWED_INFORMATION_MEASURE}.")


class _InformationMeasure:
    """A wrapper class used for the calculation the result of information measure between the discrete reference
    distributions of predicted and reference sentences. The class also handles input validation for `alpha` and
    `beta` parameters.

    Args:
        information_measure:
            A name of information measure to be used. Please use one of: ['kl_divergence', 'alpha_divergence',
            'beta_divergence', 'ab_divergence', 'renyi_divergence', 'l1_distance', 'l2_distance', 'l_infinity_distance',
            'fisher_rao_distance']
        alpha:
            Alpha parameter of the divergence used for alpha, AB and Rényi divergence measures.
        beta:
            Beta parameter of the divergence used for beta and AB divergence measures.

    Raises:
        ValueError:
            If information measure is one from alpha, AB or Rényi divergence and parameter `alpha` is `None`.
        ValueError:
            If information measure is one from beta or divergence and parameter `beta` is `None`.
        ValueError:
            If information measure is alpha divergence and parameter `alpha` equals 0 or 1.
        ValueError:
            If information measure is beta divergence and parameter `beta` equals 0 or -
        ValueError:
            If information measure is AB divergence and parameter `alpha`, `beta` or `alpha + beta` equal 0.
        ValueError:
            If information measure is Rényi divergence and parameter `alpha` equals 1.
    """

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
        if self.information_measure == _IMEnum.RENYI_DIVERGENCE and alpha == 1:
            raise ValueError(f"Parameter `alpha` is expected to be differened from 1 for {information_measure}.")

        self.alpha = alpha
        self.beta = beta

    def __call__(self, preds_distribution: Tensor, target_distribtuion: Tensor) -> Tensor:
        information_measure_function = getattr(self, f"_calculate_{self.information_measure}")
        return information_measure_function(preds_distribution, target_distribtuion)

    @staticmethod
    def _calculate_kl_divergence(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate Kullback-Leibler divergence between discrete distributions of predicted and reference
        sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Kullback-Leibler divergence between discrete distributions of predicted and reference sentences.
        """
        return torch.sum(preds_distribution * torch.log(preds_distribution / target_distribution), dim=-1)

    def _calculate_alpha_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate alpha divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Alpha divergence between discrete distributions of predicted and reference sentences.
        """
        _alpha_denom = self.alpha * (self.alpha - 1)
        alpha_divergence = (
            1 - torch.sum(target_distribution ** self.alpha * preds_distribution ** (1 - self.alpha), dim=-1)
        ) / _alpha_denom
        return alpha_divergence

    def _calculate_ab_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate AB divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            AB divergence between discrete distributions of predicted and reference sentences.
        """
        x = torch.log(torch.sum(target_distribution ** (self.beta + self.alpha), dim=-1))
        y = torch.log(torch.sum(preds_distribution ** (self.beta + self.alpha), dim=-1))
        z = torch.log(torch.sum(target_distribution ** self.alpha * preds_distribution ** self.beta, dim=-1))
        ab_divergence = (
            x / (self.beta * (self.beta + self.alpha)) + y / (self.beta + self.alpha) - z / (self.alpha * self.beta)
        )
        return ab_divergence

    def _calculate_beta_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate beta divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Beta divergence between discrete distributions of predicted and reference sentences.
        """
        self.alpha = 1.0
        beta_divergence = self._calculate_ab_divergence(preds_distribution, target_distribution)
        return beta_divergence

    def _calculate_renyi_divergence(self, preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate Rényi divergence between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Rényi divergence between discrete distributions of predicted and reference sentences.
        """
        renyi_divergence = (
            torch.log(torch.sum(target_distribution ** self.alpha * preds_distribution ** (1 - self.alpha), dim=-1))
        ) / (self.alpha - 1)
        return renyi_divergence

    @staticmethod
    def _calculate_l1_distance(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate L1 distance between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            L1 distance between discrete distributions of predicted and reference sentences.
        """
        return torch.norm(target_distribution - preds_distribution, p=1, dim=-1)

    @staticmethod
    def _calculate_l2_distance(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate L2 distance between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            L2 distance between discrete distributions of predicted and reference sentences.
        """
        return torch.norm(target_distribution - preds_distribution, p=2, dim=-1)

    @staticmethod
    def _calculate_l_infinity_distance(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate L-infinity distance between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            L-infinity distance between discrete distributions of predicted and reference sentences.
        """
        return torch.norm(target_distribution - preds_distribution, p=float("inf"), dim=-1)

    @staticmethod
    def _calculate_fisher_rao_distance(preds_distribution: Tensor, target_distribution: Tensor) -> Tensor:
        """Calculate Fisher-Rao distance between discrete distributions of predicted and reference sentences.

        Args:
            preds_distribution:
                Discrete reference distribution of predicted sentences over the vocabulary.
            target_distribution:
                Discrete reference distribution of reference sentences over the vocabulary.

        Return:
            Fisher-Rao distance between discrete distributions of predicted and reference sentences.
        """
        return 2 * torch.acos(torch.clamp(torch.sqrt(preds_distribution * target_distribution).sum(-1), 0, 1))


def _load_tokenizer_and_model(
    model_name_or_path: Union[str, os.PathLike], device: Union[str, torch.device]
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load HuggingFace `transformers`' tokenizer and model. This function also handle a device placement.

    Args:
        model_name_or_path:
            A name or a model path used to load `transformers` pretrained model.
        device:
            A device to be used for calculation.

    Return:
        Initialized `transformers`' tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, output_hidden_states=True)
    model.eval()
    model.to(device)
    return tokenizer, model


def _infolm_update(preds: Union[str, Sequence[str]], target: Sequence[Union[str, Sequence[str]]]):
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
    Calculate `InfoLM`_ - i.e. calculate a distance/divergence between predicted and reference sentence discrete
    distribution using one of the following information measures:
        - `KL divergence`_
        - `alpha divergence`_
        - `beta divergence`_
        - `AB divergence`_
        - `Rényi divergence`_
        - L1 distance
        - L2 distance
        - L-infinity distance
        - `Fisher-Rao distance`_

    Args:
        preds:
            An iterable of hypothesis corpus.
        target:
            An iterable of iterables of reference corpus.
        model_name_or_path:
            A name or a model path used to load `transformers` pretrained model.
        temperature:
            A temperature for calibrating language modelling. For more information, please reference `InfoLM`_ paper.
        information_measure:
            A name of information measure to be used. Please use one of: ['kl_divergence', 'alpha_divergence',
            'beta_divergence', 'ab_divergence', 'renyi_divergence', 'l1_distance', 'l2_distance', 'l_infinity_distance',
            'fisher_rao_distance']
        idf:
            An indication of whether normalization using inverse document frequencies should be used.
        alpha:
            Alpha parameter of the divergence used for alpha, AB and Rényi divergence measures.
        beta:
            Beta parameter of the divergence used for beta and AB divergence measures.
        device:
            A device to be used for calculation.
    """
    tokenizer, model = _load_tokenizer_and_model(model_name_or_path, device)
    information_measure_cls = _InformationMeasure(information_measure, alpha, beta)

    _infolm_update(preds, target)

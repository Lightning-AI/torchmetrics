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
from functools import partial

import pytest
import torch

from torchmetrics.functional.text.infolm import infolm
from torchmetrics.text.infolm import InfoLM
from unittests.text.helpers import TextTester
from unittests.text.inputs import HYPOTHESIS_A, HYPOTHESIS_C, _inputs_single_reference

# Small bert model with 2 layers, 2 attention heads and hidden dim of 128
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
MAX_LENGTH = 30  # the selected model has default max_length = 20 and we have longer sequences


def reference_infolm_score(preds, target, model_name, information_measure, idf, alpha, beta, functional_test: bool):
    """Currently, a reference package is not available.

    We, therefore, are enforced to relied on hard-coded results for now.
    TODO: Add a link to public repo containing a script for generating reproducible results.
    """
    if model_name != "google/bert_uncased_L-2_H-128_A-2":
        raise ValueError(
            "`model_name` is expected to be 'google/bert_uncased_L-2_H-128_A-2' as this model was used for the result "
            "generation."
        )
    precomputed_result = {
        "kl_divergence": torch.tensor([-3.2250, -0.1784, -0.1784, -2.2182]),
        "beta_divergence": torch.tensor([0.5812, 0.0716, 0.0716, 0.3721]),
        "renyi_divergence": torch.tensor([0.4357, 0.0333, 0.0333, 0.3615]),
        "l2_distance": torch.tensor([0.2053, 0.1114, 0.1114, 0.2522]),
        "fisher_rao_distance": torch.tensor([1.5637, 0.4957, 0.4957, 1.4570]),
    }
    # Add results for idf=True -> for functional metrics, we calculate idf only over the batch yet
    if functional_test:
        precomputed_result.update(
            {
                "alpha_divergence": torch.tensor([-1.2851, -0.1262, -0.1262, -1.3096]),
                "ab_divergence": torch.tensor([5.9517, 0.5222, 0.5222, 7.0017]),
                "l1_distance": torch.tensor([0.9679, 0.1877, 0.1877, 0.9561]),
                "l_infinity_distance": torch.tensor([0.0789, 0.0869, 0.0869, 0.2324]),
            }
        )
    else:
        precomputed_result.update(
            {
                "alpha_divergence": torch.tensor([-1.2893, -0.1262, -0.1262, -1.4035]),
                "ab_divergence": torch.tensor([5.9565, 0.5222, 0.5222, 7.1950]),
                "l1_distance": torch.tensor([0.9591, 0.1877, 0.1877, 1.0823]),
                "l_infinity_distance": torch.tensor([0.0777, 0.0869, 0.0869, 0.2614]),
            }
        )

    res = precomputed_result[information_measure]
    if HYPOTHESIS_A in preds:
        res = res[:2]
    elif HYPOTHESIS_C in preds:
        res = res[2:]
    else:
        raise ValueError("Invalid example provided.")
    return res.mean()


@pytest.mark.parametrize(
    ["information_measure", "idf", "alpha", "beta"],
    [
        ("kl_divergence", False, 0.25, 0.25),
        ("alpha_divergence", True, 0.4, 0.3),
        ("beta_divergence", False, None, 0.6),
        ("ab_divergence", True, 0.25, 0.25),
        ("renyi_divergence", False, 0.3, 0.1),
        ("l1_distance", True, None, None),
        ("l2_distance", False, None, None),
        ("l_infinity_distance", True, None, None),
        ("fisher_rao_distance", False, 0.25, 0.25),
    ],
)
@pytest.mark.parametrize(
    ["preds", "targets"],
    [(_inputs_single_reference.preds, _inputs_single_reference.targets)],
)
class TestInfoLM(TextTester):
    # Set atol = 1e-4 as reference results are rounded
    atol = 1e-4

    @pytest.mark.parametrize("ddp", [False, True])
    @pytest.mark.parametrize("dist_sync_on_step", [False, True])
    def test_infolm_class(self, ddp, dist_sync_on_step, preds, targets, information_measure, idf, alpha, beta):
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "information_measure": information_measure,
            "idf": idf,
            "alpha": alpha,
            "beta": beta,
            "max_length": MAX_LENGTH,
        }
        reference_metric = partial(
            reference_infolm_score,
            model_name=MODEL_NAME,
            information_measure=information_measure,
            idf=idf,
            alpha=alpha,
            beta=beta,
            functional_test=False,
        )

        self.run_class_metric_test(
            ddp=ddp,
            preds=preds,
            targets=targets,
            metric_class=InfoLM,
            sk_metric=reference_metric,
            dist_sync_on_step=dist_sync_on_step,
            metric_args=metric_args,
        )

    def test_infolm_functional(self, preds, targets, information_measure, idf, alpha, beta):
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "information_measure": information_measure,
            "idf": idf,
            "alpha": alpha,
            "beta": beta,
            "max_length": MAX_LENGTH,
        }
        reference_metric = partial(
            reference_infolm_score,
            model_name=MODEL_NAME,
            information_measure=information_measure,
            idf=idf,
            alpha=alpha,
            beta=beta,
            functional_test=True,
        )

        self.run_functional_metric_test(
            preds,
            targets,
            metric_functional=infolm,
            sk_metric=reference_metric,
            metric_args=metric_args,
        )

    def test_infolm_differentiability(self, preds, targets, information_measure, idf, alpha, beta):
        metric_args = {
            "model_name_or_path": MODEL_NAME,
            "information_measure": information_measure,
            "idf": idf,
            "alpha": alpha,
            "beta": beta,
            "max_length": MAX_LENGTH,
        }

        self.run_differentiability_test(
            preds=preds,
            targets=targets,
            metric_module=InfoLM,
            metric_functional=infolm,
            metric_args=metric_args,
        )

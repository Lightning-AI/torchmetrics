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
from torchmetrics.functional.regression.concordance import _concordance_corrcoef_compute
from torchmetrics.regression.pearson import PearsonCorrCoef, _final_aggregation


class ConcordanceCorrCoef(PearsonCorrCoef):
    def compute(self):
        if self.mean_x.numel() > 1:  # multiple devices, need further reduction
            mean_x, mean_y, var_x, var_y, corr_xy, n_total = _final_aggregation(
                self.mean_x, self.mean_y, self.var_x, self.var_y, self.corr_xy, self.n_total
            )
        else:
            mean_x = self.mean_x
            mean_y = self.mean_y
            var_x = self.var_x
            var_y = self.var_y
            corr_xy = self.corr_xy
            n_total = self.n_total
        return _concordance_corrcoef_compute(mean_x, mean_y, var_x, var_y, corr_xy, n_total)

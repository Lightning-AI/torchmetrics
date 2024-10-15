# Copyright The Lightning team.
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

import copy
import math
import os
import warnings
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import adaptive_max_pool2d, relu, softmax
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchmetrics.utilities import rank_zero_info
from torchmetrics.utilities.imports import _LIBROSA_AVAILABLE, _REQUESTS_AVAILABLE

if _LIBROSA_AVAILABLE and _REQUESTS_AVAILABLE:
    import librosa
    import requests
else:
    librosa, requests = None, None

__doctest_requires__ = {("non_intrusive_speech_quality_assessment",): ["librosa", "requests"]}

NISQA_DIR = "~/.mbchl/NISQA"


def non_intrusive_speech_quality_assessment(x: Tensor, fs: int) -> Tensor:
    """Non-Intrusive Speech Quality Assessment (NISQA v2.0) [1], [2].

    .. note:: Using this metric requires you to have ``librosa`` and ``requests`` installed. Install as
        ``pip install librosa requests``.

    Args:
        x: float tensor with shape ``(...,time)``
        fs: sampling frequency of input

    Returns:
        Float tensor with shape ``(...,5)`` corresponding to overall MOS, noisiness, coloration, discontinuity and
        loudness

    Raises:
        ModuleNotFoundError:
            If ``librosa`` or ``requests`` are not installed
        ValueError:
            If the input is too long, causing the number of segments to exceed the maximum allowed

    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment
        >>> _ = torch.manual_seed(42)
        >>> x = torch.randn(16000)
        >>> non_intrusive_speech_quality_assessment(x, 16000)
        tensor([1.0433, 1.9545, 2.6087, 1.3460, 1.7117])

    References:
        - [1] G. Mittag and S. Möller, "Non-intrusive speech quality assessment for super-wideband speech communication
          networks", in Proc. ICASSP, 2019.
        - [2] G. Mittag, B. Naderi, A. Chehadi and S. Möller, "NISQA: A deep CNN-self-attention model for
          multidimensional speech quality prediction with crowdsourced datasets", in Proc. INTERSPEECH, 2021.

    """
    if not _LIBROSA_AVAILABLE or not _REQUESTS_AVAILABLE:
        raise ModuleNotFoundError(
            "NISQA metric requires that librosa and requests are installed. Install as `pip install librosa requests`."
        )
    model, args = _load_nisqa_model()
    model.eval()
    in_shape = x.shape
    x = x.reshape(-1, in_shape[-1])
    x = _get_librosa_melspec(x.cpu().numpy(), fs, args)
    x, n_wins = _segment_specs(torch.from_numpy(x), args)
    with torch.no_grad():
        x = model(x, n_wins.expand(x.shape[0]))
    # ["mos_pred", "noi_pred", "dis_pred", "col_pred", "loud_pred"]
    return x.reshape(in_shape[:-1] + (5,))


@lru_cache
def _load_nisqa_model() -> tuple[nn.Module, dict]:
    model_path = os.path.expanduser(os.path.join(NISQA_DIR, "nisqa.tar"))
    if not os.path.exists(model_path):
        _download_weights()
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    args = checkpoint["args"]
    model = NISQADIM(args)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, args


def _download_weights() -> None:
    url = "https://github.com/gabrielmittag/NISQA/raw/refs/heads/master/weights/nisqa.tar"
    nisqa_dir = os.path.expanduser(NISQA_DIR)
    os.makedirs(nisqa_dir, exist_ok=True)
    saveto = os.path.join(nisqa_dir, "nisqa.tar")
    if os.path.exists(saveto):
        return
    rank_zero_info(f"downloading {url} to {saveto}")
    myfile = requests.get(url)
    with open(saveto, "wb") as f:
        f.write(myfile.content)


class NISQADIM(nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.cnn = Framewise(args)
        self.time_dependency = TimeDependency(args)
        pool = Pooling(args)
        self.pool_layers = _get_clones(pool, 5)

    def forward(self, x: Tensor, n_wins: Tensor) -> Tensor:
        x = self.cnn(x, n_wins)
        x, n_wins = self.time_dependency(x, n_wins)
        out = [mod(x, n_wins) for mod in self.pool_layers]
        return torch.cat(out, dim=1)


class Framewise(nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.model = AdaptCNN(args)

    def forward(self, x: Tensor, n_wins: Tensor) -> Tensor:
        x_packed = pack_padded_sequence(x, n_wins, batch_first=True, enforce_sorted=False)
        x = self.model(x_packed.data)
        x = x_packed._replace(data=x)
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0.0, total_length=n_wins.max())
        return x


class AdaptCNN(nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.pool_1 = args["cnn_pool_1"]
        self.pool_2 = args["cnn_pool_2"]
        self.pool_3 = args["cnn_pool_3"]
        self.dropout = nn.Dropout2d(p=args["cnn_dropout"])
        cnn_pad = (1, 0) if args["cnn_kernel_size"][0] == 1 else (1, 1)
        self.conv1 = nn.Conv2d(1, args["cnn_c_out_1"], args["cnn_kernel_size"], padding=cnn_pad)
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, args["cnn_c_out_2"], args["cnn_kernel_size"], padding=cnn_pad)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, args["cnn_c_out_3"], args["cnn_kernel_size"], padding=cnn_pad)
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv4 = nn.Conv2d(self.conv3.out_channels, args["cnn_c_out_3"], args["cnn_kernel_size"], padding=cnn_pad)
        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)
        self.conv5 = nn.Conv2d(self.conv4.out_channels, args["cnn_c_out_3"], args["cnn_kernel_size"], padding=cnn_pad)
        self.bn5 = nn.BatchNorm2d(self.conv5.out_channels)
        self.conv6 = nn.Conv2d(
            self.conv5.out_channels,
            args["cnn_c_out_3"],
            (args["cnn_kernel_size"][0], args["cnn_pool_3"][1]),
            padding=(1, 0),
        )
        self.bn6 = nn.BatchNorm2d(self.conv6.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = relu(self.bn1(self.conv1(x)))
        x = adaptive_max_pool2d(x, output_size=(self.pool_1))
        x = relu(self.bn2(self.conv2(x)))
        x = adaptive_max_pool2d(x, output_size=(self.pool_2))
        x = self.dropout(x)
        x = relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = relu(self.bn4(self.conv4(x)))
        x = adaptive_max_pool2d(x, output_size=(self.pool_3))
        x = self.dropout(x)
        x = relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = relu(self.bn6(self.conv6(x)))
        return x.view(-1, self.conv6.out_channels * self.pool_3[0])


class TimeDependency(nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.model = SelfAttention(args)

    def forward(self, x: Tensor, n_wins: Tensor) -> Tensor:
        return self.model(x, n_wins)


class SelfAttention(nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        encoder_layer = SelfAttentionLayer(args)
        self.norm1 = nn.LayerNorm(args["td_sa_d_model"])
        self.linear = nn.Linear(args["cnn_c_out_3"] * args["cnn_pool_3"][0], args["td_sa_d_model"])
        self.layers = _get_clones(encoder_layer, args["td_sa_num_layers"])
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, n_wins: Tensor) -> Tensor:
        src = self.linear(src)
        output = src.transpose(1, 0)
        output = self.norm1(output)
        for mod in self.layers:
            output, n_wins = mod(output, n_wins)
        return output.transpose(1, 0), n_wins


class SelfAttentionLayer(nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(args["td_sa_d_model"], args["td_sa_nhead"], args["td_sa_dropout"])
        self.linear1 = nn.Linear(args["td_sa_d_model"], args["td_sa_h"])
        self.dropout = nn.Dropout(args["td_sa_dropout"])
        self.linear2 = nn.Linear(args["td_sa_h"], args["td_sa_d_model"])
        self.norm1 = nn.LayerNorm(args["td_sa_d_model"])
        self.norm2 = nn.LayerNorm(args["td_sa_d_model"])
        self.dropout1 = nn.Dropout(args["td_sa_dropout"])
        self.dropout2 = nn.Dropout(args["td_sa_dropout"])
        self.activation = relu

    def forward(self, src: Tensor, n_wins: Tensor) -> Tensor:
        mask = torch.arange(src.shape[0])[None, :] < n_wins[:, None]
        src2 = self.self_attn(src, src, src, key_padding_mask=~mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, n_wins


class Pooling(nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.model = PoolAttFF(args)

    def forward(self, x: Tensor, n_wins: Tensor) -> Tensor:
        return self.model(x, n_wins)


class PoolAttFF(torch.nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.linear1 = nn.Linear(args["td_sa_d_model"], args["pool_att_h"])
        self.linear2 = nn.Linear(args["pool_att_h"], 1)
        self.linear3 = nn.Linear(args["td_sa_d_model"], 1)
        self.activation = relu
        self.dropout = nn.Dropout(args["pool_att_dropout"])

    def forward(self, x: Tensor, n_wins: Tensor) -> Tensor:
        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2, 1)
        mask = torch.arange(att.shape[2])[None, :] < n_wins[:, None]
        att[~mask.unsqueeze(1)] = float("-inf")
        att = softmax(att, dim=2)
        x = torch.bmm(att, x)
        x = x.squeeze(1)
        return self.linear3(x)


def _get_librosa_melspec(y: np.ndarray, sr: int, args: dict) -> None:
    hop_length = int(sr * args["ms_hop_length"])
    win_length = int(sr * args["ms_win_length"])
    with warnings.catch_warnings():
        # ignore empty mel filter warning since this is expected when input signal is not fullband and the model
        # penalizes the signal accordingly
        # see https://github.com/gabrielmittag/NISQA/issues/6#issuecomment-838157571
        warnings.filterwarnings("ignore", message="Empty filters detected in mel frequency basis")
        melspec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            S=None,
            n_fft=args["ms_n_fft"],
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=1.0,
            n_mels=args["ms_n_mels"],
            fmin=0.0,
            fmax=args["ms_fmax"],
            htk=False,
            norm="slaney",
        )
    # batch processing of librosa.core.amplitude_to_db is not equivalent to individual processing due to top_db being
    # relative to max value
    # so process individually and then stack
    return np.stack([librosa.amplitude_to_db(m, ref=1.0, amin=1e-4, top_db=80.0) for m in melspec])


def _segment_specs(x: np.ndarray, args: dict) -> tuple[Tensor, Tensor]:
    seg_length = args["ms_seg_length"]
    seg_hop = args["ms_seg_hop_length"]
    max_length = args["ms_max_segments"]
    n_wins = x.shape[2] - (seg_length - 1)
    idx1 = torch.arange(seg_length)
    idx2 = torch.arange(n_wins)
    idx3 = idx1.unsqueeze(0) + idx2.unsqueeze(1)
    x = x.transpose(2, 1)[:, idx3, :].unsqueeze(2).transpose(4, 3)
    x = x[:, ::seg_hop]
    n_wins = math.ceil(n_wins / seg_hop)
    if max_length < n_wins:
        raise ValueError("Maximum number of melspectrogram segments exceeded. Use shorter audio.")
    x_padded = torch.zeros((x.shape[0], max_length, x.shape[2], x.shape[3], x.shape[4]))
    x_padded[:, :n_wins, :] = x
    return x_padded, torch.tensor(n_wins)


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from featurizer import MelSpectrogramConfig


def pad(y_fake, y_real):
    """ for generator output and ground truth padding"""
    sz_diff = np.abs(y_real.size(-1) - y_fake.size(-1))
    if sz_diff != 0:
        if y_real.size(-1) > y_fake.size(-1):
            return F.pad(y_fake, (0, sz_diff)), y_real
        else:
            return y_fake, F.pad(y_real, (0, sz_diff))


class ResBlock(nn.Module):
    def __init__(
            self,
            ch: int,    # number of channels remain the same
            kernel_r: int,  # k_r
            d_rb: List[List[int]],  # D_r
            leaky_relu_slope: float = 0.1
    ):
        super(ResBlock, self).__init__()
        self.version = 1 if len(d_rb[0]) == 2 else 3
        self.leaky_relu_slope = leaky_relu_slope
        self.conv1 = nn.ModuleList([
            nn.Conv1d(
                ch, ch, kernel_r,
                dilation=d_r[0], padding=self._get_padding(kernel_r, d_r[0])
            ) for d_r in d_rb
        ])
        if self.version == 1:
            self.conv2 = nn.ModuleList([
                nn.Conv1d(
                    ch, ch, kernel_r,
                    dilation=d_r[1], padding=self._get_padding(kernel_r, d_r[1])
                ) for d_r in d_rb
            ])
        # self.layers = [
        #     nn.ModuleList([
        #         nn.Conv1d(
        #             ch, ch, kernel_r,
        #             dilation=d_r, padding=self._get_padding(kernel_r, d_r)
        #         ) for d_r in d_tuple
        #     ]) for d_tuple in d_rb
        # ]

    def forward(self, x):
        for conv_tuple in zip(self.conv1, self.conv2) if self.version == 1 else self.conv1:
            if self.version == 1:
                for conv in conv_tuple:
                    out = F.leaky_relu(x, self.leaky_relu_slope)
                    out = conv(out)
            else:
                out = F.leaky_relu(x, self.leaky_relu_slope)
                out = conv_tuple(out)
            x = x + out
        return x

    def _get_padding(self, kernel_r, dilation_r):
        """padding = same for stride=1 and odd kernel"""
        return int(dilation_r * (kernel_r - 1) / 2)


class MRFModule(nn.Module):
    def __init__(
            self,
            ch: int,
            kernel_rb: List[int],    # k_r
            d_rb: List[List[List[int]]],    # D_r
            leaky_relu_slope: float = 0.1
    ):
        super(MRFModule, self).__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.resblocks = nn.ModuleList()
        for k_r, d_r in zip(kernel_rb, d_rb):
            self.resblocks.append(ResBlock(ch, k_r, d_r, leaky_relu_slope))

    def forward(self, x):
        out = None
        for resblock in self.resblocks:
            if out is None:
                out = resblock(x)
            else:
                out += resblock(x)
        return out / len(self.resblocks)


class Generator(nn.Module):
    def __init__(
            self,
            in_ch: int,
            hidden_ch: int = 512,    # h_u
            kernel_up: List[int] = [16, 16, 4, 4],    # k_u
            kernel_rb: List[int] = [3, 7, 11],  # k_r
            d_rb: List[List[List[int]]] = [
                [[1, 1], [3, 1], [5, 1]],
                [[1, 1], [3, 1], [5, 1]],
                [[1, 1], [3, 1], [5, 1]]
            ],  # D_r
            leaky_relu_slope: float = 0.1
    ):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.conv_in = nn.Conv1d(in_ch, hidden_ch, 7, dilation=1, padding=3)
        upsample_layers = []
        for l, k_u in enumerate(kernel_up, 1):
            upsample_layers.extend([
                nn.LeakyReLU(leaky_relu_slope),
                nn.ConvTranspose1d(
                    hidden_ch // (2 ** (l - 1)), hidden_ch // (2 ** l), k_u,
                    stride=k_u // 2, padding=k_u // 4
                ),
                MRFModule(hidden_ch // (2 ** l), kernel_rb, d_rb, leaky_relu_slope)
            ])
        self.upsample_layers = nn.Sequential(*upsample_layers)
        self.conv_out = nn.Conv1d(hidden_ch // (2 ** len(kernel_up)), 1, 7, padding=3)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.upsample_layers(x)
        x = F.leaky_relu(x, self.leaky_relu_slope)
        x = self.conv_out(x)
        x = torch.tanh(x)
        return x.squeeze(1)


def get_generator(version: int = 1, n_mels: int = MelSpectrogramConfig.n_mels):
    assert version in [1, 2, 3]
    if version == 1:
        return Generator(n_mels)
    if version == 2:
        return Generator(n_mels, 128)
    if version == 3:
        return Generator(n_mels, 256, [16, 16, 8], [3, 5, 7], [[[1], [2]], [[2], [6]], [[3], [12]]])


if __name__ == '__main__':
    a = torch.randn(1, 80, 400)
    # resblock = ResBlock(80, 3, [[1, 1], [3, 1], [5, 1]])
    # print(resblock)
    # print(resblock(a).shape)
    # print(dict(resblock.named_parameters()).keys())

    # mrf = MRFModule(80, [3, 7, 11], [[[1,1],[3,1],[5,1]], [[1,1],[3,1],[5,1]], [[1,1],[3,1],[5,1]]])
    # print(mrf)
    # print(dict(mrf.named_parameters()).keys())
    # print(mrf(a).shape)
    for i in [1, 2, 3]:
        try:
            model = get_generator(i, 80)
            print(dict(model.named_parameters()).keys())
            model(a)
        except:
            raise
    model = Generator(80)
    print(model)
    print(dict(model.named_parameters()).keys())
    print(model(a).shape)

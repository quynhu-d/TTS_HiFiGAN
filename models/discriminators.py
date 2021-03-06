from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm


class MSDSub(nn.Module):
    # from MelGAN
    def __init__(
            self,
            leaky_relu_slope: float = .1,
            use_spectral: bool = False
    ):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        w_norm = spectral_norm if use_spectral else weight_norm
        self.conv_block = nn.ModuleList([
            w_norm(nn.Conv1d(1, 16, 15, stride=1))
        ])
        cur_ch = 16
        cur_groups = 1
        for _ in range(4):
            self.conv_block.append(
                w_norm(nn.Conv1d(cur_ch, min(cur_ch * 4, 1024), 41, stride=4, groups=cur_groups * 4, padding=20))
            )
            cur_ch = min(cur_ch * 4, 1024)
            cur_groups *= 4
        self.conv_block.append(w_norm(nn.Conv1d(cur_ch, cur_ch, 5, stride=1, padding=2)))

        self.conv_out = w_norm(nn.Conv1d(cur_ch, 1, 3, stride=1, padding=1))

    def forward(self, x, return_ft=True):
        x = x.unsqueeze(1)
        if return_ft:
            ft_maps = []  # for feature matching
        for conv in self.conv_block:
            x = conv(x)
            x = F.leaky_relu(x, self.leaky_relu_slope)
            if return_ft:
                ft_maps.append(x)
        x = self.conv_out(x)
        if return_ft:
            ft_maps.append(x)
        if return_ft:
            return x.flatten(1, -1), ft_maps
        else:
            return x.flatten(1, -1)


class MSDiscriminator(nn.Module):
    def __init__(
            self,
            n_discs: int = 3,  # number of sub discriminators
            leaky_relu_slope: float = .1
    ):
        super(MSDiscriminator, self).__init__()
        self.disc_layers = nn.ModuleList([MSDSub(leaky_relu_slope, use_spectral=True)])
        self.disc_layers.extend([MSDSub(leaky_relu_slope) for _ in range(n_discs - 1)])
        self.pooling = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)

    def forward(self, x, return_ft=True):
        if return_ft:
            ft_maps = []
        outputs = []
        for disc in self.disc_layers:
            if return_ft:
                output, ft_map = disc(x)
            else:
                output = disc(x, return_ft)
            outputs.append(output)
            if return_ft:
                ft_maps.append(ft_map)
            x = self.pooling(x)
        if return_ft:
            return outputs, ft_maps
        else:
            return outputs


class MPDSub(nn.Module):
    def __init__(
            self,
            p: int,  # period
            leaky_relu_slope: float = .1
    ):
        super().__init__()
        self.p = p
        self.leaky_relu_slope = leaky_relu_slope

        self.conv_block = nn.ModuleList()

        cur_ch = 2 ** 5
        for l in range(1, 6):
            if l != 5:
                self.conv_block.append(
                    weight_norm(nn.Conv2d(1 if l == 1 else cur_ch, cur_ch * 2, (5, 1), (3, 1), padding=2))
                )
            else:
                self.conv_block.append(
                    weight_norm(nn.Conv2d(cur_ch, cur_ch * 2, (5, 1), 1, padding=(2, 0)))
                )
            cur_ch *= 2

        self.conv_out = weight_norm(nn.Conv2d(cur_ch, 1, (3, 1), padding=(1, 0)))

    def forward(self, x, return_ft=True):
        """ x: batch of shape N x T """
        if x.size(-1) % self.p != 0:
            # pad audio
            x = F.pad(x, (0, self.p - (x.size(-1) % self.p)), mode='reflect')
        n, t = x.size()
        x = x.unsqueeze(1)  # N x 1 x T
        x = x.view(n, 1, t // self.p, self.p)  # N x 1 x T / p x p

        if return_ft:
            ft_maps = []  # for feature matching
        for conv in self.conv_block:
            x = conv(x)
            x = F.leaky_relu(x, self.leaky_relu_slope)
            if return_ft:
                ft_maps.append(x)
        x = self.conv_out(x)
        if return_ft:
            ft_maps.append(x)
        if return_ft:
            return x.flatten(1, -1), ft_maps
        else:
            return x.flatten(1, -1)


class MPDiscriminator(nn.Module):
    def __init__(
            self,
            periods: List[int] = [2, 3, 5, 7, 11],
            leaky_relu_slope: float = .1
    ):
        super(MPDiscriminator, self).__init__()
        self.disc_layers = nn.ModuleList([
            MPDSub(p, leaky_relu_slope) for p in periods
        ])

    def forward(self, x, return_ft=True):
        if return_ft:
            ft_maps = []
        outputs = []
        for disc in self.disc_layers:
            if return_ft:
                output, ft_map = disc(x)
            else:
                output = disc(x, return_ft)
            outputs.append(output)
            if return_ft:
                ft_maps.append(ft_map)
        if return_ft:
            return outputs, ft_maps
        else:
            return outputs


if __name__ == '__main__':
    x = torch.randn(3, 400)
    print(MPDiscriminator())
    print(dict(MPDiscriminator().named_parameters()).keys())
    print(MSDiscriminator())
    print(dict(MSDiscriminator().named_parameters()).keys())
    MSDiscriminator().to('cpu')
    MPDiscriminator().to('cpu')
    output_s, ft_maps_s = MSDiscriminator()(x)
    output_p, ft_maps_p = MPDiscriminator()(x)

    output_s_ = MSDiscriminator()(x, return_ft=False)
    output_p_ = MPDiscriminator()(x, return_ft=False)
    # print(output, ft_maps)
    # print(output[0].shape, ft_maps[2][0].shape)

import torch
import torch.nn.functional as F

from featurizer import MelSpectrogram
from models import *
from data import Batch


def get_d_loss(disc, y_fake, y_real):
    y_fake = y_fake.detach()

    d_out_real = disc(y_real, return_ft=False)
    d_out_fake = disc(y_fake, return_ft=False)

    d_adv = 0
    d_real = 0
    d_fake = 0
    for d_r, d_f in zip(d_out_real, d_out_fake):
        d_r = torch.mean((d_r - 1) ** 2)
        d_f = torch.mean(d_f ** 2)
        d_real = d_real + d_r.item()
        d_fake = d_fake + d_f.item()

        d_adv = d_adv + d_r + d_f

    return d_adv, {'real loss': d_real, 'fake_loss': d_fake}


def get_g_loss(disc, y_fake, y_real, lambda_fm=2):
    d_out_real, d_ft_real = disc(y_real)
    d_out_fake, d_ft_fake = disc(y_fake)

    g_adv = 0
    for d_out_f in d_out_fake:
        g_adv = g_adv + torch.mean((d_out_f - 1) ** 2)

    g_fm = 0
    for ft_real_, ft_fake_ in zip(d_ft_real, d_ft_fake):
        for ft_r, ft_f in zip(ft_real_, ft_fake_):
            g_fm = g_fm + F.l1_loss(ft_f, ft_r, reduction='mean')

    g_loss = g_adv + lambda_fm * g_fm
    return g_loss, {'g_adv': g_adv.item(), 'g_fm': g_fm.item()}


def get_mel_loss(y_fake, y_real, featurizer):
    return F.l1_loss(featurizer(y_fake), featurizer(y_real))


if __name__ == '__main__':
    batch = Batch(waveform=torch.randn(3, 8192), waveform_length=torch.tensor([100, 100, 100]), transcript=['', '', ''])
    featurizer = MelSpectrogram()
    batch.mel = featurizer(batch.waveform)
    gen = Generator(80, 512)
    mpd = MPDiscriminator()
    msd = MSDiscriminator()

    y_fake = gen(batch.mel)
    y_fake, y_real = pad(y_fake, batch.waveform)
    print(get_d_loss(mpd, y_fake, y_real))
    print(get_g_loss(mpd, y_fake, y_real))
    print(get_mel_loss(y_fake, y_real, featurizer))

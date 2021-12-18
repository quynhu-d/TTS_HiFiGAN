import torch
import torch.nn.functional as F


def get_d_loss(gen, disc, batch, device='cpu'):
    y_fake = gen(batch.mel).detach()

    # padding
    sz_diff = np.abs(batch.waveform.size(-1) - y_fake.size(-1))
    if sz_diff != 0:
        if batch.waveform.size(-1) > y_fake.size(-1):
            y_fake = F.pad(y_fake, (0, sz_diff))
            y_real = batch.waveform
        else:
            y_real = F.pad(batch.waveform, (0, sz_diff))
    # assert y_real.shape == y_fake.shape

    d_out_real = disc(y_real, return_ft=False)
    d_out_fake = disc(y_fake, return_ft=False)

    d_adv = 0
    d_real = 0
    d_fake = 0
    for d_r, d_f in zip(d_out_real, d_out_fake):
        d_r = torch.mean((d_r - 1) ** 2)
        d_f = torch.mean(d_f ** 2)
        d_real += d_r.item()
        d_fake += d_f.item()

        d_adv += d_r + d_f

    return d_adv, {'real loss': d_real, 'fake_loss': d_fake}


def get_g_loss(gen, disc, batch, featurizer, lambda_mel=45, lambda_fm=2, device='cpu'):
    y_fake = gen(batch.mel)

    # padding
    sz_diff = np.abs(batch.waveform.size(-1) - y_fake.size(-1))
    if sz_diff != 0:
        if batch.waveform.size(-1) > y_fake.size(-1):
            y_fake = F.pad(y_fake, (0, sz_diff))
            y_real = batch.waveform
        else:
            y_real = F.pad(batch.waveform, (0, sz_diff))
    # assert y_real.shape == y_fake.shape

    d_out_real, d_ft_real = disc(y_real)
    d_out_fake, d_ft_fake = disc(y_fake)

    g_adv = 0
    for d_out_f in d_out_fake:
        g_adv += torch.mean((d_out_f - 1) ** 2)

    g_fm = 0
    for ft_real_, ft_fake_ in zip(d_ft_real, d_ft_fake):
        for ft_r, ft_f in zip(ft_real_, ft_fake_):
            g_fm += F.l1_loss(ft_f, ft_r, reduction='mean')

    g_mel = F.l1_loss(featurizer(y_fake), featurizer(y_real))

    g_loss = g_adv + lambda_mel * g_mel + lambda_fm * g_fm
    return g_loss, {'g_adv': g_adv.item(), 'g_mel': g_mel.item(), 'g_fm': g_fm.item()}

import errno
import itertools
import os
from typing import Callable

import wandb
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm.auto import trange, tqdm

from data import *
from featurizer import *
from loss import get_d_loss, get_g_loss, get_mel_loss
from models import Generator, MPDiscriminator, MSDiscriminator, pad, get_generator
from trainer.train_config import TrainConfig
from trainer.utils import plot_spectrogram_to_buf


def train(
        train_config: TrainConfig,
        mel_config: MelSpectrogramConfig = MelSpectrogramConfig(),
        mel_aug: Callable = None,
        opt_g=None,
        opt_d=None,
        sch_g=None,
        sch_d=None,
        logging: bool = True
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        train_dataloader = DataLoader(
            LJSpeechDataset(train_config.lj_path), batch_size=train_config.batch_size, collate_fn=LJSpeechCollator()
        )
    except errno:
        raise "No dataset found at %s" % train_config.lj_path

    if not os.path.exists(train_config.save_dir):
        os.mkdir(train_config.save_dir)
    import time
    model_path = train_config.save_dir + '/%s/' % (time.strftime("%d-%m-%I-%M-%S"))
    os.mkdir(model_path)
    print('Saving model at %s' % model_path)

    featurizer = MelSpectrogram(mel_config).to(device)

    gen = get_generator(train_config.gen_version, mel_config.n_mels).to(device)
    if train_config.model_cp_path is not None and (train_config.last_epoch != -1):
        gen.load_state_dict(torch.load(train_config.model_cp_path + 'gen.pth', device))
    gen.train()
    print('Generator model', gen, sep='\n')

    mpd = MPDiscriminator().to(device)
    if train_config.model_cp_path is not None and (train_config.last_epoch != -1):
        mpd.load_state_dict(torch.load(train_config.model_cp_path + 'mpd.pth', device))
    mpd.train()
    print('MPD model', mpd, sep='\n')
    msd = MSDiscriminator().to(device)
    if train_config.model_cp_path is not None and (train_config.last_epoch != -1):
        msd.load_state_dict(torch.load(train_config.model_cp_path + 'msd.pth', device))
    msd.train()
    print('MSD model', msd, sep='\n')

    if opt_g is None:
        opt_g = torch.optim.AdamW(gen.parameters(), 2e-4, (.8, .99), weight_decay=.01)
    if train_config.model_cp_path is not None and (train_config.last_epoch != -1):
        opt_g.load_state_dict(torch.load(train_config.model_cp_path + 'opt_g.pth', device))
    if sch_g is None:
        sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=.999, last_epoch=train_config.last_epoch)

    d_params = [mpd.parameters(), msd.parameters()]
    d_params = itertools.chain(*d_params)
    if opt_d is None:
        opt_d = torch.optim.AdamW(d_params, 2e-4, (.8, .99), weight_decay=.01)
    if train_config.model_cp_path is not None and (train_config.last_epoch != -1):
        opt_d.load_state_dict(torch.load(train_config.model_cp_path + 'opt_d.pth', device))
    if sch_d is None:
        sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=.999, last_epoch=train_config.last_epoch)

    total_config = {**train_config.__dict__, **mel_config.__dict__}
    if logging:
        wandb.init(project=train_config.wandb_project, name=train_config.wandb_name, config=total_config)

    for epoch in trange(train_config.n_epochs):
        for step, batch in tqdm(
                enumerate(train_dataloader), total=len(train_dataloader),
                leave=False, desc='EPOCH %d' % epoch
        ):
            batch.to(device)
            if mel_aug is not None:
                batch.mel = mel_aug(featurizer(batch.waveform)).to(device)
            else:
                batch.mel = featurizer(batch.waveform).to(device)
            # Discriminators
            y_fake = gen(batch.mel)
            y_fake, y_real = pad(y_fake, batch.waveform)
            y_fake, y_real = y_fake.to(device), y_real.to(device)

            opt_d.zero_grad()
            # MPD
            mpd.requires_grad = True
            mpd_loss, mpd_losses = get_d_loss(mpd, y_fake, y_real)

            # MSD
            msd.requires_grad = True
            msd_loss, msd_losses = get_d_loss(msd, y_fake, y_real)

            d_loss = mpd_loss + msd_loss
            d_loss.backward()
            opt_d.step()

            # Generator
            y_fake = gen(batch.mel)
            y_fake, y_real = pad(y_fake, batch.waveform)
            y_fake, y_real = y_fake.to(device), y_real.to(device)
            opt_g.zero_grad()

            # MPD
            mpd.requires_grad = False
            g_mpd_loss, g_mpd_losses = get_g_loss(mpd, y_fake, y_real)

            # MSD
            msd.requires_grad = False
            g_msd_loss, g_msd_losses = get_g_loss(msd, y_fake, y_real)

            g_mel_loss = get_mel_loss(y_fake, y_real, featurizer)

            g_loss = g_mpd_loss + g_msd_loss + g_mel_loss * 45
            g_loss.backward()
            opt_g.step()

            if logging and (step % train_config.display_step == 0):
                mel_fake = featurizer(y_fake)
                idx = np.random.randint(batch.mel.shape[0])
                buf_true = plot_spectrogram_to_buf(batch.mel[idx].detach().cpu().numpy())
                buf_pred = plot_spectrogram_to_buf(mel_fake[idx].detach().cpu().numpy())
                wandb_mel_real = ToTensor()(Image.open(buf_true))
                wandb_mel_fake = ToTensor()(Image.open(buf_pred))
                del buf_pred
                del buf_true
                wandb.log({
                    'g_loss': g_loss,

                    'mel_loss': g_mel_loss,
                    'g_adv_loss': g_mpd_losses['g_adv'] + g_msd_losses['g_adv'],
                    'g_fm_loss': g_mpd_losses['g_fm'] + g_msd_losses['g_fm'],

                    'g_mpd_adv_loss': g_mpd_losses['g_adv'],
                    'g_mpd_fm_loss': g_mpd_losses['g_fm'],

                    'g_msd_adv_loss': g_msd_losses['g_adv'],
                    'g_msd_fm_loss': g_msd_losses['g_fm'],

                    'd_loss': d_loss.item(),

                    'mpd_loss': mpd_loss.item(),
                    'mpd_real': mpd_losses['real_loss'],
                    'mpd_fake': mpd_losses['fake_loss'],

                    'msd_loss': msd_loss.item(),
                    'msd_real': msd_losses['real_loss'],
                    'msd_fake': msd_losses['fake_loss'],

                    'mel': wandb.Image(wandb_mel_real),
                    'mel_pred': wandb.Image(wandb_mel_fake),
                    'audio': wandb.Audio(
                        batch.waveform[idx].detach().cpu().numpy(), sample_rate=MelSpectrogramConfig.sr
                    ),
                    'audio_pred': wandb.Audio(y_fake[idx].detach().cpu().numpy(), sample_rate=MelSpectrogramConfig.sr),
                    'step': step
                })

            if step % 100 == 0:
                # MODEL SAVING
                torch.save(gen.state_dict(), model_path + '/gen.pth')
                torch.save(mpd.state_dict(), model_path + '/mpd.pth')
                torch.save(msd.state_dict(), model_path + '/msd.pth')
                torch.save(opt_g.state_dict(), model_path + '/opt_g.pth')
                torch.save(opt_d.state_dict(), model_path + '/opt_d.pth')

            if train_config.overfit:
                break
        sch_g.step()
        sch_d.step()

    torch.save(gen.state_dict(), model_path + '/gen.pth')
    torch.save(mpd.state_dict(), model_path + '/mpd.pth')
    torch.save(msd.state_dict(), model_path + '/msd.pth')
    torch.save(opt_g.state_dict(), model_path + '/opt_g.pth')
    torch.save(opt_d.state_dict(), model_path + '/opt_d.pth')
    return gen, mpd, msd

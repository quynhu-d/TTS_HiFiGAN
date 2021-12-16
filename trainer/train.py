import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from data import LJSpeechDataset, LJSpeechCollator
from featurizer import MelSpectrogramConfig, MelSpectrogram
from models import *
import errno
import os
from typing import Tuple
from tqdm.auto import tqdm, trange


def train(
        mel_config: MelSpectrogramConfig,
        lj_path: str,
        n_epochs: int = 6000,
        logging: bool = False,
        overfit: bool = False,
        model_cp_path: str = None,
        wandb_resume: Tuple[str, str] = (None, None)    # resume mode and run id
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Training on', device)
    try:
        train_dataloader = DataLoader(
            LJSpeechDataset(lj_path), batch_size=16, collate_fn=LJSpeechCollator()
        )
    except errno:
        raise "No dataset found at %s" % lj_path

    # if not os.path.exists(train_config.save_dir):
    #     os.mkdir(train_config.save_dir)
    # import time
    # model_path = train_config.save_dir + '/%s/' % (time.strftime("%d-%m-%I-%M-%S"))
    # os.mkdir(model_path)
    # print('Saving model at %s' % model_path)
    model = Generator(80, 128).to(device)
    model.train()
    if overfit:
        batch = next(iter(train_dataloader))
    print(model)
    opt_g = torch.optim.AdamW(model.parameters(), 2e-4, (.8, .99), weight_decay=.01)
    sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=.999, last_epoch=-1)
    if logging:
        wandb.init(project='TTS_HiFiGAN', name='overfit_generator')
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
    for i in trange(n_epochs):
        for batch_ in train_dataloader:  # fixed batch above
            if not overfit:
                batch = batch_
            batch.to(device)
            opt_g.zero_grad()
            batch.mel = featurizer(batch.waveform).to(device)
            pred_wav = model(batch.mel)
            # print(pred_wav.size(), batch.mel.size())
            # print(featurizer(pred_wav).shape)

            sz_diff = np.abs(batch.waveform.size(-1) - pred_wav.size(-1))
            if sz_diff != 0:
                if batch.waveform.size(-1) > pred_wav.size(-1):
                    # print('padding waveform')
                    pred_wav = F.pad(pred_wav, (0, sz_diff))
                    true_wav = batch.waveform
                else:
                    # print('padding batch')
                    true_wav = F.pad(batch.waveform, (0, sz_diff))
                # batch.mel = featurizer(batch.waveform)
            # print(batch.waveform.shape, pred_wav.shape)
            assert true_wav.shape == pred_wav.shape
            pred_mel = featurizer(pred_wav)
            mel_loss = F.l1_loss(pred_mel, featurizer(true_wav)) * 45
            mel_loss.backward()
            opt_g.step()

            idx = np.random.randint(batch.mel.shape[0])
            if logging:
                wandb.log({
                    'mel_loss': mel_loss,
                    'mel': wandb.Image(batch.mel[idx]),
                    'mel_pred': wandb.Image(pred_mel[idx]),
                    'audio': wandb.Audio(batch.waveform[idx].detach().cpu().numpy(),
                                         sample_rate=MelSpectrogramConfig.sr),
                    'audio_pred': wandb.Audio(pred_wav[idx].detach().cpu().numpy(),
                                              sample_rate=MelSpectrogramConfig.sr),
                    'step': i
                })
            if overfit:
                break
        sch_g.step()
    return model

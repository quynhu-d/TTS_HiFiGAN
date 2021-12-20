import argparse
import os

import torch
import torchaudio

from featurizer import MelSpectrogram, MelSpectrogramConfig
from models import Generator, get_generator


def test(
        model_path: str,
        wav_dir: str,    # directory of test files
        model_version: int = 1,
        mel_config: MelSpectrogramConfig = MelSpectrogramConfig(),
        test_save_dir: str = None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = get_generator(model_version, mel_config.n_mels)
    gen.load_state_dict(torch.load(model_path, device))
    gen.eval()

    featurizer = MelSpectrogram(mel_config)

    wavs = []
    wav_lengths = []
    mels = []

    reconstructed_wavs = []
    reconstructed_mels = []
    wav_files = os.listdir(wav_dir)
    for wav_file in wav_files:
        if wav_file.find('.wav') == -1:
            continue
        wavs.append(torchaudio.load(wav_dir + wav_file)[0])
        wav_lengths.append(len(wavs[-1]))
        mels.append(featurizer(wavs[-1]))
        with torch.no_grad():
            reconstructed_wavs.append(gen(mels[-1]))
        reconstructed_mels.append(featurizer(reconstructed_wavs[-1]).squeeze(0).detach().cpu().numpy())
        if test_save_dir is not None:
            if not os.path.exists(test_save_dir):
                os.mkdir(test_save_dir)
            torchaudio.save(test_save_dir + wav_file, reconstructed_wavs[-1], sample_rate=mel_config.sr)
    return (reconstructed_wavs, reconstructed_mels), mels


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        "-m",
        "--model",
        default=None,
        type=str,
        help="generator model path (default: None)",
    )
    args.add_argument(
        "-v",
        "--version",
        default=1,
        type=int,
        help="generator model version (default: 1)",
    )
    args.add_argument(
        "-d",
        "--data",
        default=None,
        type=str,
        help="path to data to test (default: None)",
    )
    args.add_argument(
        "-o",
        "--out",
        default=None,
        type=str,
        help="dir to save reconstructed wavs at (default: None)",
    )

    arg_dict = args.parse_args().__dict__
    model_path = arg_dict['model']
    wav_dir = arg_dict['data']
    test_save_dir = arg_dict['out']
    model_version = arg_dict['version']
    # transcript = [
    # "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
    # "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
    # "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
    # ]
    test(model_path, wav_dir, model_version=model_version, test_save_dir=test_save_dir)

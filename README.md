# TTS HiFiGAN
> This repository includes HiFiGAN model implementation.

## Directory Layout
    .
    ├── augmentations           # spectrogram augmentations
    ├── data                    # dataset and collator for LJ
    ├── featurizer              # mel spectrograms
    ├── models                  # HiFiGAN (Generator and MP and MS Discriminators)
    ├── trainer                 # training functions
    ├── test.py                 # test function
    └── requirements.txt

## Logs
For training and testing runs see [`Colab Notebook`](https://colab.research.google.com/drive/1F6MeixSW1Nx8H0jWTWHpeUnFhJho4XaU?usp=sharing).  WandB logs located in [project](https://wandb.ai/quynhu_d/TTS_HiFiGAN?workspace=user-quynhu_d).

## Cloning
    !git clone https://github.com/quynhu-d/TTS_HiFiGAN
    %cd TTS_HiFiGAN
    !pip install -r ./requirements

## Data
    !wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    !tar -xjf LJSpeech-1.1.tar.bz2 -C OUT_PATH
> Remember to pass `OUT_PATH` to `lj_path` in `TrainConfig()`.

## Training
Training is performed with `train.py` from trainer directory, configurations can be set with `TrainConfig`.

Optional parameters:
- mel spectrogram config: MelSpectrogramConfig object
- spectrogram augmentations: can also be passed (for examples see [`Colab notebook`](https://colab.research.google.com/drive/1F6MeixSW1Nx8H0jWTWHpeUnFhJho4XaU?usp=sharing))
- PyTorch optimizers and schedulers (by default AdamW and ExponentialLR are set as in paper)
- logging: set to False to disable (wandb) logging


    from featurizer import MelSpectrogramConfig
    from trainer import TrainConfig, train

    train_config = TrainConfig()
    mel_config = MelSpectrogramConfig()
    train(train_config, mel_config)
> To continue training from checkpoint, define `model_cp_path` and set `last_epoch` in TrainConfig.

## Inference
For inference use `test.py`. 

    !python test.py -m MODEL_PATH -v 1 -d TEST_DATA_DIR -o SAVE_DIR
_________________________
    -m MODEL_PATH    - generator model path
    -v 1             - generator version (1, 2 or 3)
    -d TEST_DATA_DIR - directory of data to test
    -o SAVE_DIR      - (optional) directory to save reconstructed wavs at

The function returns reconstructed wavs, their spectrograms and original spectrograms.

    import test
    (r_wavs, r_mels), mels = test.test(
        '/content/drive/MyDrive/TTS_HiFiGAN/saved/20-12-11-10-36/gen.pth',
        '/content/drive/MyDrive/TTS_HiFiGAN/test_data/'
    )
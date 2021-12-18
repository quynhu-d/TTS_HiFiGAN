from dataclasses import dataclass


@dataclass
class TrainConfig:
    wandb_project: str = 'TTS_HiFiGAN'
    wandb_name: str = 'default'
    display_step: int = 100    # audio logging step

    lj_path: str = '.'
    batch_size: int = 16
    val_split: float = None

    n_epochs: int = 50
    lr: float = 3e-4
    save_dir: str = 'saved/'

    last_epoch: int = -1
    model_cp_path: str = None
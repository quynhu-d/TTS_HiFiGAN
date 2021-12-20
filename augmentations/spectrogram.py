from typing import List, Callable

import numpy as np
import torch
import torchaudio
from torch import Tensor

from augmentations.base import AugmentationBase


class SequentialAugmentation(AugmentationBase):
    def __init__(self, augmentation_list: List[Callable]):
        self.augmentation_list = augmentation_list

    def __call__(self, data: Tensor) -> Tensor:
        x = data
        for augmentation in self.augmentation_list:
            x = augmentation(x)
        return x


class RandomApply:
    def __init__(self, augmentation: Callable, p: float = .5):
        assert 0 <= p <= 1
        self.augmentation = augmentation
        self.p = p

    def __call__(self, data: Tensor) -> Tensor:
        if np.random.random() < self.p:
            return self.augmentation(data)
        else:
            return data


class FrequencyMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.FrequencyMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)


class TimeMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.TimeMasking(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)


class TimeStretch(AugmentationBase):
    def __init__(self, min_rate, max_rate, *args, **kwargs):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self._aug = torchaudio.transforms.TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        rate = np.random.uniform(self.min_rate, self.max_rate)    # choose random rate from given range
        x = data.type(torch.complex128)
        return self._aug(x, rate).type(torch.float64)


if __name__ == '__main__':
    spec = torch.randn(3, 80, 400)
    print(TimeStretch(n_freq=80, min_rate=.8, max_rate=1.2)(spec).shape)
    print(TimeMasking(50)(spec))

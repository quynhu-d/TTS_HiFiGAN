import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from typing import List


class MSDiscriminator(nn.Module):
    def __init__(self):
        super(MSDiscriminator, self).__init__()

    def forward(self, x):
        pass


class MPDiscriminator(nn.Module):
    def __init__(self):
        super(MPDiscriminator, self).__init__()

    def forward(self, x):
        pass

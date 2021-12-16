from dataclasses import dataclass
from typing import Optional
from typing import Tuple, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    transcript: List[str]
    mel: torch.Tensor = None
    # tokens: torch.Tensor
    # token_lengths: torch.Tensor
    # durations: Optional[torch.Tensor] = None
    # duration_preds: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'Batch':
        self.waveform = self.waveform.to(device)
        self.waveform_length = self.waveform_length.to(device)
        if self.mel is not None:
            self.mel = self.mel.to(device)
        # self.tokens = self.tokens.to(device)
        # self.token_lengths = self.token_lengths.to(device)
        # if self.durations is not None:
        #     self.durations = self.durations.to(device)
        # if self.duration_preds is not None:
        #     self.duration_preds = self.duration_preds.to(device)


class LJSpeechCollator:
    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveform_length, transcript, segment_sz = list(
            zip(*instances)
        )
        # TODO: change segment size pass
        # waveform = pad_sequence([
        #     waveform_[0] for waveform_ in waveform
        # ]).transpose(0, 1)
        # waveform_length = torch.cat(waveform_length)

        new_waveforms = []
        for waveform_, segment_size in zip(waveform, segment_sz):
            waveform_ = waveform_.squeeze(0)
            assert len(waveform_.size()) == 1
            if waveform_.size(0) >= segment_size:
                max_audio_start = waveform_.size(0) - segment_size
                audio_start = np.random.randint(0, max_audio_start)
                waveform_ = waveform_[audio_start:audio_start+segment_size]
            else:
                waveform_ = torch.nn.functional.pad(waveform_, (0, segment_size - waveform_.size(0)), 'constant')
            new_waveforms.append(waveform_)
        waveform = torch.stack(new_waveforms)
        waveform_length = torch.cat(waveform_length)

        # tokens = pad_sequence([
        #     tokens_[0] for tokens_ in tokens
        # ]).transpose(0, 1)
        # token_lengths = torch.cat(token_lengths)

        return Batch(waveform, waveform_length, transcript)

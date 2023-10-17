import torchaudio
from torch import nn
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.TimeStretch()
        self._hop_length = args.hop_length

    def __call__(self, data: Tensor):
        return self._aug(data, self._hop_length)

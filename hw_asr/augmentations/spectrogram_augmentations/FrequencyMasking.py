import torchaudio
from torch import nn
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class FrequencyMasking(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(*args, **kwargs)
        )

    def __call__(self, data: Tensor):
        return self._aug(data)
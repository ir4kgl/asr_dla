import torch
import logging
from typing import List

from torch.nn import ConstantPad2d

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    result_batch["text"] = list(x["text"] for x in dataset_items)
    result_batch["text_encoded"] = list(
        x["text_encoded"] for x in dataset_items)
    result_batch["duration"] = list(x["duration"] for x in dataset_items)
    result_batch["audio_path"] = list(x["audio_path"] for x in dataset_items)

    pad_audio = max(list(x["audio"].shape[-1] for x in dataset_items))
    pad_spec = max(list(x["spectrogram"].shape[-1] for x in dataset_items))

    result_batch["audio"] = torch.stack(
        tuple(ConstantPad2d((0, pad_audio - x["audio"].shape[-1], 0, 0), 0)(x["audio"]) for x in dataset_items))
    result_batch["spectrogram"] = torch.stack(
        tuple(ConstantPad2d((0, pad_spec - x["spectrogram"].shape[-1], 0, 0), 0)(x["spectrogram"]) for x in dataset_items))

    return result_batch

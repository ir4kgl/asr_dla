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

    result_batch["text_encoded_length"] = max(
        list(x["text_encoded"].shape[-1] for x in dataset_items))
    result_batch["spectrogram_length"] = max(
        list(x["spectrogram"].shape[-1] for x in dataset_items))

    result_batch["text_encoded"] = torch.cat(
        tuple(ConstantPad2d((0, result_batch["text_encoded_length"] - x["text_encoded"].shape[-1], 0, 0), 0)(x["text_encoded"]) for x in dataset_items))

    result_batch["spectrogram"] = torch.cat(
        tuple(ConstantPad2d((0,  result_batch["spectrogram_length"] - x["spectrogram"].shape[-1], 0, 0), 0)(x["spectrogram"]) for x in dataset_items))

    result_batch["text_encoded_length"] = torch.full(
        (len(dataset_items),), result_batch["text_encoded_length"], dtype=torch.long)
    result_batch["spectrogram_length"] = torch.full(
        (len(dataset_items),), result_batch["spectrogram_length"], dtype=torch.long)
    return result_batch

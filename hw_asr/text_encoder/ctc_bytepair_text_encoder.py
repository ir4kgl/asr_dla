from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


from pyctcdecode import build_ctcdecoder
import multiprocessing

from tokenizers import Tokenizer


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCBPE(CharTextEncoder):
    EMPTY_TOK = ''

    def __init__(self, tokenizer_path, lm_path=None, alpha=0.5, beta=1.0):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        alphabet = sorted(self.tokenizer.get_vocab().keys())
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.decoder = build_ctcdecoder(
            vocab, kenlm_model_path=lm_path, alpha=alpha, beta=beta)

    def ctc_decode(self, inds: List[int]) -> str:
        res = list()
        PREV = 0
        for i in inds:
            if i == 0 or i == PREV:
                PREV = i
                continue
            res.append(self.ind2char[i])
            PREV = i
        return ''.join(res)

    def ctc_beam_search(self, probs, probs_length, beam_size):

        logits_list = [p[:p_len]for p, p_len in zip(probs.cpu().detach().numpy(), probs_length.numpy())]
        with multiprocessing.get_context("fork").Pool(20) as pool:
            text_list = self.decoder.decode_batch(
                pool, logits_list, beam_width=beam_size)
        return text_list

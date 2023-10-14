from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    last_is_empty: bool


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

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

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        hypos = {}
        zero_h = Hypothesis('', False)
        hypos[zero_h] = 1.

        for i in range(probs_length):
            new_hypos = defaultdict(float)

            for next_ind, next_char in self.ind2char.items():
                for prefix in hypos.keys():
                    p = hypos[prefix] * probs[i][next_ind]
                    last_is_empty = (next_char == self.EMPTY_TOK)
                    if last_is_empty:
                        new_text = prefix.text
                    elif len(prefix.text) >= 1 and prefix.text[-1] == next_char and (not prefix.last_is_empty):
                        new_text = prefix.text
                    else:
                        new_text = prefix.text + next_char
                    h = Hypothesis(new_text, last_is_empty)
                    new_hypos[h] += p

            hypos = dict(sorted(new_hypos.items(), key=lambda x: x[1],
                         reverse=True)[:beam_size])

        return sorted(hypos.items(), key=lambda x: x[1], reverse=True)[:beam_size]

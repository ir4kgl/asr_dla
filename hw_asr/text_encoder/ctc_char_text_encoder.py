from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


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
        hypos: List[Hypothesis] = []

        hypos.append(Hypothesis('', 1.))

        for i in range(char_length):
            new_hypos: List[Hypothesis] = []

            for next_ind, next_char in self.ind2char.items():
                for prefix in hypos:
                    p = prefix.prob * probs[i][next_ind]
                    if len(prefix.text) > 0 and prefix.text[-1] == next_char:
                        new_hypos.append((prefix.text, p))
                    elif len(prefix.text) > 0 and prefix.text[-1] == self.EMPTY_TOK:
                        new_hypos.append((prefix.text[:-1] + next_char, p))
                    else:
                        new_hypos.append((prefix.text + next_char, p))
            hypos = list(sorted(new_hypos, key=lambda x: x.prob, reverse=True)[:beam_size])

        return sorted(hypos, key=lambda x: x.prob, reverse=True)

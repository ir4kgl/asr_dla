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
                        beam_size: int = 5) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        for ind, char in self.ind2char.items():
            hypos.append(Hypothesis(char, probs[0][ind]))

        hypos = sorted(hypos, key=lambda x: x.prob, reverse=True)

        for i in range(1, probs_length):
            new_hypos = defaultdict(float)
            hypos = hypos[:beam_size]
            for old_hypo in hypos:
                for ind, char in self.ind2char.items():
                    if old_hypo.text[-1] == self.EMPTY_TOK or old_hypo.text[-1] == char:
                        text = old_hypo.text[:-1] + char
                    else:
                        text = old_hypo.text + char
                    if i == probs_length-1 and text[-1] == self.EMPTY_TOK:
                        text = text[:-1]
                    prob = old_hypo.prob * probs[i][ind]
                    new_hypos[text] += prob

            hypos = [Hypothesis(*x) for x in sorted(
                new_hypos.items(), key=lambda x: x[1], reverse=True)]

        return hypos

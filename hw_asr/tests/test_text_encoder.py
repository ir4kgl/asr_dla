import unittest
import numpy as np

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d " \
               "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        probs = np.array(
            [[0.140, 0.391, 0.197, 0.271],
             [0.257, 0.096, 0.341, 0.305],
             [0.248, 0.402, 0.267, 0.083],
             [0.149, 0.336, 0.358, 0.197],]
        )

        encoder = CTCCharTextEncoder("ABC")

        all_hypos = encoder.ctc_beam_search(probs, 4, 3)
        self.assertEqual(len(all_hypos), 8)
        all_hypos = dict(all_hypos)
        self.assertAlmostEqual(all_hypos["AB"], 0.031651708842)
        self.assertAlmostEqual(all_hypos["ABA"], 0.046971825486)
        self.assertAlmostEqual(all_hypos["ABAB"], 0.019188464196)
        self.assertAlmostEqual(all_hypos["ABAC"], 0.010559015214)
        self.assertAlmostEqual(all_hypos["ABC"], 0.012298592982)
        self.assertAlmostEqual(all_hypos["ACA"], 0.02325114735)
        self.assertAlmostEqual(all_hypos["ACAB"], 0.01716270258)
        self.assertAlmostEqual(all_hypos["ACAC"], 0.00944428047)

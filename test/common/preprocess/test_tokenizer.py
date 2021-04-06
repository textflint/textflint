import unittest

from textflint.common.preprocess.tokenizer import *

sents = [
        "we're playing ping pang ball, you are so lazy. She's so beautiful!",
        'I dont know this issue.',
        "2020-year-plan is to write 3 papers"
    ]


class TestTokenizer(unittest.TestCase):
    def test_sentence_tokenize(self):
        self.assertRaises(AssertionError, sentence_tokenize, sents)
        self.assertEqual(2, len(sentence_tokenize(sents[0])))
        self.assertEqual(["we're playing ping pang ball, you "
                          "are so lazy.", "She's so beautiful!"],
                         sentence_tokenize(sents[0]))

    def test_tokenize_one_sent(self):
        self.assertRaises(AssertionError, tokenize_one_sent, sents)
        self.assertEqual(sents[0].split(' '), tokenize_one_sent(
            sents[0], split_by_space=True))
        self.assertEqual(['we', "'re", 'playing', 'ping', 'pang', 'ball', ',',
                          'you', 'are', 'so', 'lazy', '.', 'She', "'s", 'so',
                          'beautiful', '!'], tokenize_one_sent(sents[0]))

    def test_tokenize_and_untokenize(self):
        self.assertRaises(AssertionError, tokenize, sents)
        self.assertRaises(AssertionError, untokenize, sents[0])

        for sent in sents:
            self.assertEqual(sent, untokenize(tokenize(sent)))

        self.assertEqual("I don't know this 'issue,.",
                         untokenize(tokenize("I don't know this `issue,.")))


if __name__ == "__main__":
    unittest.main()

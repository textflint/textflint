import unittest

from textflint.common.preprocess.en_processor import *

sents = [
        "we're playing ping pang ball, you are so lazy. She's so beautiful!",
        'I dont know this issue.',
        "2020-year-plan is to write 3 papers"
    ]


class TestTokenizer(unittest.TestCase):
    Processor = EnProcessor()
    def test_sentence_tokenize(self):
        self.assertRaises(AssertionError, self.Processor.sentence_tokenize, sents)
        self.assertEqual(2, len(self.Processor.sentence_tokenize(sents[0])))
        self.assertEqual(["we're playing ping pang ball, you "
                          "are so lazy.", "She's so beautiful!"],
                         self.Processor.sentence_tokenize(sents[0]))

    def test_tokenize_one_sent(self):
        self.assertRaises(AssertionError,
                          self.Processor.tokenize_one_sent, sents)
        self.assertEqual(sents[0].split(' '), self.Processor.tokenize_one_sent(
            sents[0], split_by_space=True))
        self.assertEqual(['we', "'re", 'playing', 'ping', 'pang', 'ball', ',',
                          'you', 'are', 'so', 'lazy', '.', 'She', "'s", 'so',
                          'beautiful', '!'],
                         self.Processor.tokenize_one_sent(sents[0]))

    def test_tokenize_and_untokenize(self):
        self.assertRaises(AssertionError,
                          self.Processor.tokenize, sents)
        self.assertRaises(AssertionError,
                          self.Processor.inverse_tokenize, sents[0])

        for sent in sents:
            self.assertEqual(
                sent,
                self.Processor.inverse_tokenize(self.Processor.tokenize(sent)))

        self.assertEqual("I don't know this 'issue,.",
                         self.Processor.inverse_tokenize(
                             self.Processor.tokenize(
                                 "I don't know this `issue,.")))


if __name__ == "__main__":
    unittest.main()

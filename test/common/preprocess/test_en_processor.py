import unittest

from textflint.common.preprocess.en_processor import EnProcessor


class TestEnProcessor(unittest.TestCase):
    test_processor = EnProcessor()

    def test_word_tokenize(self):
        self.assertRaises(AssertionError, self.test_processor.word_tokenize, [])
        self.assertEqual([], self.test_processor.word_tokenize(''))
        sents = [
            "we're playing ping pang ball, you are so lazy. She's so beautiful!",
            'I dont know this issue.',
            "I don't know this `issue,.",
            "2020-year-plan is to write 3 papers",
            'this is ., token ? ! @# . Ok!'
        ]
        for sent in sents:
            self.assertEqual(self.test_processor.word_tokenize(
                sent, is_one_sent=1), self.test_processor.word_tokenize(
                sent, is_one_sent=0))
            self.assertEqual(self.test_processor.word_tokenize(
                sent, is_one_sent=1, split_by_space=1), sent.split(' '))
            self.assertEqual(self.test_processor.word_tokenize(
                sent, is_one_sent=0, split_by_space=1), sent.split(' '))

    def test_get_pos(self):
        self.assertRaises(AssertionError, self.test_processor.get_pos, {})
        self.assertEqual([], self.test_processor.get_pos(''))
        self.assertEqual([('All', 'DT'), ('things', 'NNS'), ('in', 'IN'),
                          ('their', 'PRP$'), ('being', 'NN'), ('are', 'VBP'),
                          ('good', 'JJ'), ('for', 'IN'), ('something', 'NN'),
                          ('.', '.')], self.test_processor.get_pos(
            'All things in their being are good for something.'))

    def test_get_ner(self):
        self.assertRaises(AssertionError, self.test_processor.get_pos, {})
        self.assertEqual([], self.test_processor.get_pos(''))
        sent = 'Lionel Messi is a football player from Argentina.'
        self.assertEqual([('Lionel Messi', 0, 12, 'PERSON'),
                          ('Argentina', 39, 48, 'LOCATION')],
                        self.test_processor.get_ner(sent, return_char_idx=True))
        self.assertEqual([('Lionel Messi', 0, 2, 'PERSON'),
                          ('Argentina', 7, 8, 'LOCATION')],
                       self.test_processor.get_ner(sent, return_char_idx=False))

    def test_get_dep_parser(self):
        self.assertRaises(ValueError, self.test_processor.get_dep_parser, {})
        self.assertEqual([], self.test_processor.get_dep_parser(''))
        sent = 'The quick brown fox jumps over the lazy dog.'
        self.assertEqual([('The', 'DT', 4, 'det'), ('quick', 'JJ', 4, 'amod'),
                          ('brown', 'JJ', 4, 'amod'), ('fox', 'NN', 5, 'nsubj'),
                          ('jumps', 'VBZ', 0, 'ROOT'), ('over', 'IN', 5, 'prep'),
                          ('the', 'DT', 9, 'det'), ('lazy', 'JJ', 9, 'amod'),
                          ('dog', 'NN', 6, 'pobj'), ('.', '.', 5, 'punct')],
                         self.test_processor.get_dep_parser(sent))

    def test_get_cfg_parser(self):
        self.assertRaises(ValueError, self.test_processor.get_parser, {})
        self.assertEqual('', self.test_processor.get_parser(''))
        sent = 'Lionel Messi is a football player from Argentina.'
        self.assertEqual('(ROOT\n  (S\n    (NP (NNP Lionel) (NNP Messi))\n'
                         '    (VP\n      (VBZ is)\n      (NP\n        '
                         '(NP (DT a) (NN football) (NN player))\n        '
                         '(PP (IN from) (NP (NNP Argentina)))))\n    (. .)))',
                         self.test_processor.get_parser(sent))

    def test_get_lemmas(self):
        sent = 'The quick brown fox jumps over the lazy dog .'
        self.assertEqual(['the', 'quick', 'brown', 'fox', 'jump', 'over',
                          'the', 'lazy', 'dog', '.'],
                         self.test_processor.get_lemmas(
                             [(i, i) for i in sent.split(' ')]))

    # TODO test other fiction


if __name__ == "__main__":
    unittest.main()

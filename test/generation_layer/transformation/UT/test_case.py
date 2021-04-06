import unittest

from textflint.input_layer.component.sample import SASample
from textflint.generation_layer.transformation.UT.word_case import WordCase


class TestWordWordCase(unittest.TestCase):
    def test_transformation(self):
        sent1 = 'The quick brown fox jumps over the lazy dog .'
        data_sample = SASample({'x': sent1, 'y': "negative"})

        # test wrong model
        self.assertRaises(ValueError, WordCase, 'random little')

        # test model
        test_case = WordCase()
        self.assertTrue(test_case.case_type in ['upper', 'lower', 'title',
                                                'random'])

        # test lower
        self.assertEqual(
            [word.lower() for word in data_sample.get_words('x')],
            WordCase('lower').transform(data_sample)[0].get_words('x'))

        # test upper
        self.assertEqual(
            [word.upper() for word in data_sample.get_words('x')],
            WordCase('upper').transform(data_sample)[0].get_words('x'))

        # test title
        self.assertEqual(
            [word.title() for word in data_sample.get_words('x')],
            WordCase('title').transform(data_sample)[0].get_words('x'))

        # test special case
        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual('', WordCase('lower').transform(
            special_sample)[0].get_text('x'))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual('~!@#$%^7890"\'', WordCase('lower').transform(
            special_sample)[0].get_text('x'))


if __name__ == "__main__":
    unittest.main()

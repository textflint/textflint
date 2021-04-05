import unittest

from TextFlint.input_layer.component.sample.sa_sample import *
from TextFlint.generation_layer.validator.sentence_encoding \
    import SentenceEncodings
from TextFlint.generation_layer.validator.max_words_perturbed \
    import MaxWordsPerturbed
from TextFlint.generation_layer.validator.edit_distance \
    import EditDistance
from TextFlint.generation_layer.validator.translate_score import TranslateScore
from TextFlint.generation_layer.validator.gpt2_perplexity import GPT2Perplexity
from TextFlint.input_layer.dataset.dataset import Dataset


class TestValidator(unittest.TestCase):
    def test_validator(self):
        # Create test data
        trans_sentences = ['There is a book on the desk .',
                           'There is a book on the floor .',
                           'There is a cookie on the desk .',
                           'There is a desk on the book .',
                           'There desk a on the is a book .']
        ori_sentence = 'There is a book on the desk .'
        ori_dataset = Dataset('SA')
        trans_dataset = Dataset('SA')

        ori_dataset.append(SASample({'x': ori_sentence, 'y': '1'}), sample_id=0)
        for trans_sentence in trans_sentences:
            trans_dataset.append(
                SASample({'x': trans_sentence, 'y': '1'}), sample_id=0)

        # test declutr encoder
        score = SentenceEncodings(ori_dataset, trans_dataset, 'x').score
        real_score = [1.0, 0.934, 0.844, 0.975, 0.880]
        for i, j in zip(score, real_score):
            self.assertAlmostEqual(i, j, 3)

        # test max words perturbed
        score = MaxWordsPerturbed(ori_dataset, trans_dataset, 'x').score
        real_score = [1.0, 0.875, 0.875, 0.75, 0.625]
        for i, j in zip(score, real_score):
            self.assertAlmostEqual(i, j, 3)

        # test levenshtein distance
        score = EditDistance(ori_dataset, trans_dataset, 'x').score
        real_score = [1.0, 0.828, 0.897, 0.793, 0.448]
        for i, j in zip(score, real_score):
            self.assertAlmostEqual(i, j, 3)

        # test translate score
        score = TranslateScore(ori_dataset, trans_dataset, 'x', 'bleu').score
        real_score = [1.0, 0.707, 0.5, 0.0, 0.0]
        for i, j in zip(score, real_score):
            self.assertAlmostEqual(i, j, 3)

        score = TranslateScore(ori_dataset, trans_dataset, 'x', 'chrf').score
        real_score = [1.0, 0.745, 0.696, 0.617, 0.638]
        for i, j in zip(score, real_score):
            self.assertAlmostEqual(i, j, 3)

        score = TranslateScore(ori_dataset, trans_dataset, 'x', 'meteor').score
        real_score = [0.999, 0.865, 0.865, 0.878, 0.867]
        for i, j in zip(score, real_score):
            self.assertAlmostEqual(i, j, 3)

        # test gtp 2
        score = GPT2Perplexity(ori_dataset, trans_dataset, 'x').score
        real_score = [1.0, 1.0, 1.0, 0.591, 0.139]
        for i, j in zip(score, real_score):
            self.assertAlmostEqual(i, j, 3)


if __name__ == '__main__':
    unittest.main()

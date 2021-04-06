r"""
BLEU Constraints
=====================================
"""

import nltk
import nltk.translate.chrf_score

from .validator import Validator
from TextFlint.common.preprocess.tokenizer import tokenize
__all__ = ['TranslateScore']


class TranslateScore(Validator):
    def __init__(
        self,
        origin_dataset,
        trans_dataset,
        fields,
        type
    ):
        r"""
        A constraint on Translate score difference.

        :param ~TextFlint.input_layer.dataset origin_dataset:
            the dataset of origin sample
        :param ~TextFlint.input_layer.dataset trans_dataset:
            the dataset of translate sample
        :param str|list fields: the name of the origin field need compare.
        :param str type: the type of the scoring index.

        """
        super().__init__(
            origin_dataset,
            trans_dataset,
            fields
        )
        if type not in ['bleu', 'chrf', 'meteor']:
            raise ValueError('Please choose type from '
                             'bleu, chrf, meteor,not {0}'.format(type))
        self.type = type

    def __repr__(self):
        return "TranslateScore" + '-' + self.type

    def validate(self, transformed_text, reference_text):
        r"""
        Calculate the score

        :param str transformed_text: transformed sentence
        :param str reference_text: origin sentence
        :return float: the score of two sentence

        """
        if self.type == 'bleu':
            return nltk.translate.bleu_score.sentence_bleu(
                [tokenize(reference_text)], tokenize(transformed_text))
        elif self.type == 'chrf':
            return nltk.translate.chrf_score.sentence_chrf(
                tokenize(reference_text), tokenize(transformed_text))
        elif self.type == 'meteor':
            return nltk.translate.meteor([reference_text], transformed_text)

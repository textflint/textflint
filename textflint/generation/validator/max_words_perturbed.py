r"""
Max Perturb Words Constraints
=====================================
"""

import numpy
from .validator import Validator
__all__ = ['MaxWordsPerturbed']


class MaxWordsPerturbed(Validator):
    r"""
    A constraint representing a maximum allowed perturbed words.
        We use the lcs div the long of the sentence as the score.

    :param ~textflint.input.dataset origin_dataset:
            the dataset of origin sample
    :param ~textflint.input.dataset trans_dataset:
        the dataset of translate sample
    :param str|list fields: the name of the origin field need compare.
    :param bool need_tokens: if we need tokenize the sentence

    """
    def __init__(
            self,
            origin_dataset,
            trans_dataset,
            fields,
            need_tokens=True
    ):
        super().__init__(
            origin_dataset,
            trans_dataset,
            fields,
            need_tokens=need_tokens
        )

    def __repr__(self):
        return "MaxWordsPerturbed"

    def validate(self, transformed_text, reference_text):
        r"""
        Calculate the score

        :param str transformed_text: transformed sentence
        :param str reference_text: origin sentence
        :return float: the score of two sentence

        """
        num_words_diff = self.get_lcs(reference_text, transformed_text)

        return num_words_diff / len(reference_text)

    @staticmethod
    def get_lcs(token1, token2):
        """
        Calculating the longest common subsequence

        :param list token1: the first token list
        :param list token2: the second token list
        :return int: the longest common subsequence

        """
        l1 = len(token1)
        l2 = len(token2)
        lcs = numpy.zeros([l1 + 1, l2 + 1])
        for i in range(0, l1):
            for j in range(0, l2):
                if token1[i] == token2[j]:
                    lcs[i + 1][j + 1] = lcs[i][j] + 1
                else:
                    lcs[i + 1][j + 1] = max(lcs[i + 1][j], lcs[i][j + 1])
        return lcs[l1][l2]

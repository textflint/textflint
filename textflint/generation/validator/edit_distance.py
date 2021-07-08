r"""
Levenshtein distance class
=====================================
"""
import editdistance

from .validator import Validator
__all__ = ['EditDistance']


class EditDistance(Validator):
    r"""
    A constraint on edit distance (Levenshtein Distance).
    We use the Levenshtein Distance div the long of the sentence as score.

    :param ~textflint.input.dataset origin_dataset:
            the dataset of origin sample
    :param ~textflint.input.dataset trans_dataset:
        the dataset of translate sample
    :param str|list fields: the name of the origin field need compare.

    """
    def __init__(
        self,
        origin_dataset,
        trans_dataset,
        fields
    ):
        super().__init__(
            origin_dataset,
            trans_dataset,
            fields
        )

    def __repr__(self):
        return "EditDistance"

    def validate(self, transformed_text, reference_text):
        r"""
        Calculate the score

        :param str transformed_text: transformed sentence
        :param str reference_text: origin sentence
        :return float: the score of two sentence

        """
        dis = editdistance.eval(transformed_text, reference_text)
        return 1 - min(dis, len(reference_text)) / len(reference_text)

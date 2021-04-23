r"""
Transforms an input by replacing its number word
==========================================================
"""

from ..transformation import Transformation
from ....common.utils.num_word import _get_contradictory_hypothesis

__all__ = ['NumWord']

LOWER_YEAR_NUM = 1000
UPPER_YEAR_NUM = 2020


class NumWord(Transformation):
    r"""
    Transforms an input by replacing its number word
    exmaple:
    {
        hypothesis: Mr Zhang has more than 20 students in Fudan university.
        premise: Mr Zhang has 10 students in Fudan university.
        y: contradicition
    }
    """

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'NumWord'

    def transform(self, sample, n=1, **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~NLISample sample: Data sample for augmentation
        :param int n: Default is 1. MAX number of unique augmented output
        :param **kwargs:
        :return: Augmented data
        """
        transform_results = self._transform(sample, **kwargs)

        if transform_results:
            return [data for data in transform_results if not data.is_origin]
        else:
            return []

    def _transform(self, sample, n=1,  **kwargs):
        r"""
        Transform text string, this kind of transformation can only produce one
        sample.

        :param ~NLISample sample: input data, a NLISample contains 'hypothesis'
            field, 'premise' field and 'y' field
        :param int n: number of generated samples, this transformation can only
            generate one sample
        :return list trans_samples: transformed sample list that only contain
            one sample
        """
        tokens = sample.get_words('premise')
        flag = False

        for num, token in enumerate(tokens):
            if token.isdigit():
                number = int(token)
                if LOWER_YEAR_NUM <= number <= UPPER_YEAR_NUM:
                    continue
                cont_hyp = _get_contradictory_hypothesis(tokens, num, number)
                flag = True
                break

        if not flag:
            return None

        sample = sample.replace_fields(
            ['premise', 'y'], [cont_hyp, 'contradiction']
        )

        return [sample]


r"""
Transforms an input by replacing its number word
==========================================================
"""

from ..transformation import Transformation
from ....common.utils.num_word import _get_contradictory_hypothesis

LOWER_YEAR_NUM = 1000
UPPER_YEAR_NUM = 2020

__all__ = ['SwapNum']


class SwapNum(Transformation):
    r"""
    Transforms an input by replacing its number word

    Exmaple::

    {
        sentence1: Every people has two hands.
        sentence2: There are five hands with all of us.
        label: 0
    }

    """

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'SwapNum'

    def transform(self, sample, n=1, **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~SMSample sample: Data sample for augmentation
        :param int n: Default is 1. MAX number of unique augmented output
        :param **kwargs:
        :return: Augmented data

        """
        transform_results = self._transform(sample, **kwargs)

        if transform_results:
            return [data for data in transform_results if not data.is_origin]
        else:
            return []

    def _transform(self, sample, **kwargs):
        r"""
        Transform text string, this kind of transformation can only produce one sample.

        :param ~NLISample sample: input data, a NLISample contains
            'sentence1' field, 'sentence2' field and 'y' field
        :param int n: number of generated samples,
            this transformation can only generate one sample
        :return list trans_samples: transformed sample list
            that only contain one sample

        """
        tokens = sample.get_words('sentence2')
        flag = False

        for num, token in enumerate(tokens):
            if token.isdigit():
                number = int(token)
                if LOWER_YEAR_NUM <= number <= UPPER_YEAR_NUM:
                    continue
                # ent_hyp = _get_entailed_hypothesis(tokens, num, number)
                cont_hyp = _get_contradictory_hypothesis(tokens, num, number)
                flag = True
                break

        if not flag:
            return None

        sample = sample.replace_fields(['sentence2', 'y'], [cont_hyp, '0'])

        return [sample]

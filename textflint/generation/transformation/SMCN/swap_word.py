r"""
Getting antonym of word
==========================================================
"""
from textflint.generation.transformation.UTCN.cn_get_antonym import CnAntonym
from textflint.input.component.sample.utcn_sample import UTCnSample
from ..transformation import Transformation
LOWER_YEAR_NUM = 1000
UPPER_YEAR_NUM = 2020

__all__ = ['SwapWord']


class SwapWord(Transformation):
    r"""
    Transforms an input by replacing its antonym

    Exmaple::

    {
        sentence1: 我喜欢这本书。
        sentence2: 这本书是我讨厌的。
        label: 0
    }

    """

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'SwapWord'

    def transform(self, sample, n=1, **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~SMCNSample sample: Data sample for augmentation
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

        :param ~SMCNSample sample: input data, a NLISample contains
            'sentence1' field, 'sentence2' field and 'y' field
        :param int n: number of generated samples,
            this transformation can only generate one sample
        :return list trans_samples: transformed sample list
            that only contain one sample

        """
        label_tag = sample.get_value('y')

        if label_tag != '1':
            return None
        tokens = sample.get_words('sentence2')
        tokens="".join(tokens)
        cnantonym=CnAntonym()
        sample1 = UTCnSample({'x': tokens})
        sentence2 = cnantonym._transform(sample1)
        if len(sentence2)>0:
            sentence2 = sentence2[0].get_tokens('x')
            sentence2 = "".join(sentence2)
            newtokens=sentence2
        else:
            return None


        sample = sample.replace_fields(['sentence2', 'y'], [newtokens, '0'])

        return [sample]

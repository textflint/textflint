r"""
Coref - Rnd concat: Concat randomly chosen samples from
    `other_samples` behind samples from `sample`
============================================

"""

import random


from ..transformation import Transformation
from ....input.component.sample import CorefSample
__all__ = ['RndConcat']


class RndConcat(Transformation):
    r""" 
    Concatenate one extra sample to the original sample, with maintaining
        the coref-relations themselves.

    Attributes:
        processor: textflint.common.preprocess.TextProcessor.

    Example::

        ori: {
            'sentences': [
                ['I', 'came'], ['I', 'saw'], ['I', 'conquered'], 
                ['Anna', 'bel', 'wanna', 'sleep'],
                ['Anna', 'bel', 'is', 'happy']
                ],
            'clusters': [
                [[1, 1], [3, 3], [5, 5]], 
                [[7, 8], [11, 12]]]}
        trans: {
            'sentences': [
                ['I', 'came'], ['I', 'saw'], ['I', 'conquered'], 
                ['Anna', 'bel', 'wanna', 'sleep'],
                ['Anna', 'bel', 'is', 'happy'],
                ['who', 'is', 'this', 'boy'], ['he', 'is', 'Jotion']],
            'clusters': [
                [[1, 1], [3, 3], [5, 5]], 
                [[7, 8], [11, 12]], 
                [[17, 18], [19, 19], [21, 21]]]}

    """

    def __init__(self, **kwargs):
        super().__init__()

    def __repr__(self):
        return 'RndConcat'

    def _transform(self, sample, n=5, **kwargs):
        r"""
        :param ~textflint.CorefSample sample: a CorefSample
        :param str|list fields: Not used
        :param int n: optional; number of generated samples
        :param list samples_other: optional, list of dict
            `samples_other` contains some other CorefSamples that also
            originate from conll-style dicts.
        :return list: samples_tfed, transformed sample list.
        """
        if sample.num_sentences() == 0: return [sample] * n
        samples_other = kwargs['samples_other']
        samples_tfed = []
        for i in range(n):
            # randomly choose a sample from samples_other
            j = int(random.random() * len(samples_other))
            # get the tfed sample and append to list
            sample_tfed = CorefSample.concat_conlls(sample, samples_other[j])
            samples_tfed.append(sample_tfed)
        return samples_tfed

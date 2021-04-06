r"""
Coref - Rnd delete: For one sample, randomly delete some sentences
    of it
==========================================================
"""

import random

from ..transformation import Transformation
__all__ = ['RndDelete']


class RndDelete(Transformation):
    r""" 
    Randomly delete trans_p * num_sentences sentences

    Attributes:
        trans_p: proportion of deleted sentences; default 0.2
        processor: textflint.common.preprocess.TextProcessor.

    Example::

        ori: {
            'sentences': [
                ['I', 'came'], ['I', 'saw'], ['I', 'conquered'], 
                ['Anna', 'bel', 'wanna', 'sleep'],
                ['Anna', 'bel', 'is', 'happy']],
            'clusters': [
                [[1, 1], [3, 3], [5, 5]], 
                [[7, 8], [11, 12]]]}
        trans: {
            'sentences': [
                ['I', 'came'], ['I', 'saw'], 
                ['Anna', 'bel', 'wanna', 'sleep'],
                ['Anna', 'bel', 'is', 'happy']],
            'clusters': [
                [[1, 1], [3, 3]], 
                [[5, 6], [9, 10]]]}

    """

    def __init__(self, trans_p=0.2, **kwargs):
        super().__init__()
        self.trans_p = trans_p

    def __repr__(self):
        return 'RndDelete'

    def _transform(self, sample, n=5, **kwargs):
        r"""
        :param ~textflint.CorefSample sample: a CorefSample
        :param str|list fields: Not used
        :param int n: optional; number of generated samples
        :return list: samples_tfed, transformed sample list.

        """
        if sample.num_sentences() <= 1: return [sample] * n
        num_sentences = sample.num_sentences()
        samples_tfed = []
        for i in range(n):
            # randomly choose sentences to preserve
            preserved_sen_idxs = []
            for j in range(num_sentences):
                if random.random() > self.trans_p:
                    preserved_sen_idxs.append(j)
            # at least preserve 1 sen; at least delete 1 sen
            if len(preserved_sen_idxs) == 0:
                preserved_sen_idxs = [0]
            if len(preserved_sen_idxs) == num_sentences:
                j = int(random.random() * num_sentences)
                preserved_sen_idxs = preserved_sen_idxs[:j] + \
                    preserved_sen_idxs[j + 1:]
            # get the tfed sample
            sample_tfed_part = sample.part_conll(preserved_sen_idxs)
            # post process: remove invalid clusters
            sample_tfed = sample_tfed_part.remove_invalid_corefs_from_part()
            # append to list
            samples_tfed.append(sample_tfed)
        return samples_tfed

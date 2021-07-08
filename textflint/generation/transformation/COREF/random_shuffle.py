r"""
Coref - Rnd shuffle: Randomly shuffle some sentences.
At least (1-2*trans_p) sentences would be at the right pos, so don't worry
==========================================================
"""

from math import ceil
import random

from ..transformation import Transformation
__all__ = ['RndShuffle']


class RndShuffle(Transformation):
    r""" 
    Randomly change the position of trans_p * num_sentences pairs of sentences

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
                ['I', 'came'], ['I', 'saw'], ['Anna', 'bel', 'wanna', 'sleep'], 
                ['I', 'conquered'], ['Anna', 'bel', 'is', 'happy']],
            'clusters': [
                [[1, 1], [3, 3], [9, 9]], 
                [[5, 6], [11, 12]]]}

    """

    def __init__(self, trans_p=0.2, **kwargs):
        super().__init__()
        self.trans_p = trans_p

    def __repr__(self):
        return 'RndShuffle'

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
        if sample.num_sentences() <= 1:
            return [sample] * n
        num_sentences = sample.num_sentences()
        samples_tfed = []

        for i in range(n):
            # shuffle: swap for trans_p * num_sentences (at least 1) times
            tfed_sen_idxs = list(range(num_sentences))
            for j in range(ceil(self.trans_p * num_sentences)):
                # randomly choose two sens ii & jj; then swap
                ii = int(random.random() * num_sentences)
                jj = int(random.random() * num_sentences)
                tmp = tfed_sen_idxs[ii]
                tfed_sen_idxs[ii] = tfed_sen_idxs[jj]
                tfed_sen_idxs[jj] = tmp
            # get the tfed sample and append to list
            sample_tfed = sample.shuffle_conll(tfed_sen_idxs)
            samples_tfed.append(sample_tfed)

        return samples_tfed

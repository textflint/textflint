r"""
Coref - Rnd Replace: some irrelevance sentences will replace the
    original sentences, the corefs including in which will be ignored.
==========================================================
"""

from copy import copy
from math import ceil
import random

from ..transformation import Transformation
__all__ = ['RndReplace']


class RndReplace(Transformation):
    r""" 
    RndReplace: trans_p * num_sentences of sentences are replaced by
        irrelevant sentences from samples_other, and the attached corefs
        will be ignored.

    Attributes:
        trans_p: proportion of inserted sentences; default 0.2
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
                ['I', 'came'], ['It', 'was', 'a', 'good', 'trip'], 
                ['I', 'conquered'], ['Anna', 'bel', 'wanna', 'sleep'], 
                ['Anna', 'bel', 'is', 'happy']],
            'clusters': [
                [[1, 1], [8, 8]], 
                [[10, 11], [16, 17]]]}

    """

    def __init__(self, trans_p=0.2, **kwargs):
        super().__init__()
        self.trans_p = trans_p

    def __repr__(self):
        return 'RndReplace'

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
        if sample.num_sentences() <= 1: return [sample] * n
        samples_other = kwargs['samples_other']
        num_sentences = sample.num_sentences()
        num_clusters = len(sample.clusters)
        samples_tfed = []
        for i in range(n):
            sample_tfed = copy(sample)
            # replace times: trans_p * num_sentences; at least 1
            for j in range(ceil(num_sentences * self.trans_p)):
                # randomly choose the irrelevant sentence
                k = int(random.random() * len(samples_other))
                sample_other = samples_other[k]
                if sample_other.num_sentences() > 0:
                    k_sen_idx = int(random.random() *
                                    sample_other.num_sentences())
                    k_sen = sample_other.get_kth_sen(k_sen_idx)
                else:
                    k_sen = ['UNK']

                # randomly choose tfed_sen_idx
                # k_sen will replace position tfed_sen_idx sentence
                # tfed_sen_idx in [1, num_sentences - 1):
                # tfed_sen_idx cannot be the first/last one
                # 1. insert after sen tfed_sen_idx
                # 2. delete the original part of sen tfed_sen_idx
                assert sample_tfed.num_sentences() == num_sentences, \
                    "Assert failed in RndReplace: " \
                    "document length does not match."
                tfed_sen_idx = int(random.random() * (num_sentences - 2)) + 1
                if tfed_sen_idx >= num_sentences: 
                    tfed_sen_idx = num_sentences - 1
                sen_start = sum(sample_tfed.sen_map[:tfed_sen_idx])
                sen_stop = sen_start + sample_tfed.sen_map[tfed_sen_idx]
                insert_at_idx = sen_stop - 1
                delete_span = [sen_start, sen_stop]
                sample_tfed = sample_tfed.insert_field_after_indices(
                    'x', [insert_at_idx], [k_sen])
                sample_tfed = sample_tfed.delete_field_at_indices(
                    'x', [delete_span])
            # get the tfed sample and append to list
            samples_tfed.append(sample_tfed)

        return samples_tfed

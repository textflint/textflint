r"""
Coref - Rnd Insert: some irrelevance sentences will be inserted
    into original sentences. These sentences include no corefs.
==========================================================
"""

from copy import copy
from math import ceil
import random

from ...transformation import Transformation
__all__ = ['RndInsert']


class RndInsert(Transformation):
    r""" 
    RndInsert: trans_p * num_sentences of irrelevance sentences are
        sampled from samples_other, and they will be inserted into original
        sentences in sample. These inserted sentences include no corefs.

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
                ['I', 'came'], ['I', 'saw'], ['who', 'is', 'this', 'boy'], 
                ['I', 'conquered'], ['Anna', 'bel', 'wanna', 'sleep'], 
                ['Anna', 'bel', 'is', 'happy']],
            'clusters': [
                [[1, 1], [3, 3], [9, 9]], 
                [[11, 12], [17, 18]]]}

    """

    def __init__(self, trans_p=0.2, **kwargs):
        super().__init__()
        self.trans_p = trans_p

    def __repr__(self):
        return 'RndInsert'

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
        num_sentences = sample.num_sentences()
        num_clusters = len(sample.clusters)
        samples_tfed = []

        for i in range(n):
            sample_tfed = copy(sample)
            # insert times: trans_p * num_sentences; at least 1
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
                # randomly choose tfed_sen_idx:
                # k_sen will be inserted before sentence tfed_sen_idx
                # tfed_sen_idx in [1, num_sentences + j):
                # tfed_sen_idx cannot be the last one
                assert sample_tfed.num_sentences() == num_sentences + j, \
                    "Assert failed in RndInsert: " \
                    "document length does not match."
                tfed_sen_idx = 1 + int(random.random() * (num_sentences + j - 1))
                insert_at_idx = sum(sample_tfed.sen_map[:tfed_sen_idx])
                sen_map = sample_tfed.sen_map

                if insert_at_idx == sum(sen_map): 
                    # i have to do this to avoid bug in common.utils.list_op
                    sample_tfed = sample_tfed.insert_field_after_indices(
                        'x', [insert_at_idx-1], [k_sen])
                    assert sen_map[-1] + len(k_sen) \
                        == sample_tfed.sen_map[-1], \
                        "Assert failed in RndInsert: sentence lengths " \
                        "seem to be unexpected after insert. " \
                        "Original sen_map: {0}, current sen_map: {1}"\
                            .format(sen_map, sample_tfed.sen_map)
                    sample_tfed.sen_map[-1] -= len(k_sen)
                    sample_tfed.sen_map.append(len(k_sen))
                else:
                    # true main logic
                    sample_tfed = sample_tfed.insert_field_before_indices(
                        'x', [insert_at_idx], [k_sen])
                    assert sen_map[tfed_sen_idx] + len(k_sen) \
                        == sample_tfed.sen_map[tfed_sen_idx], \
                        "Assert failed in RndInsert: sentence lengths " \
                        "seem to be unexpected after insert. " \
                        "Original sen_map: {0}, current sen_map: {1}"\
                            .format(sen_map, sample_tfed.sen_map)
                    sample_tfed.sen_map = sample_tfed.sen_map[:tfed_sen_idx] \
                        + [len(k_sen), sample_tfed.sen_map[tfed_sen_idx]
                           - len(k_sen)] \
                        + sample_tfed.sen_map[tfed_sen_idx+1:]
            # get the tfed sample and append to list
            samples_tfed.append(sample_tfed)

        return samples_tfed

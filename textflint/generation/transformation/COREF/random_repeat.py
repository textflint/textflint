r"""
Coref - Rnd repeat: Randomly choose some sentences, and each of them
    will be repeated somewhere else in the sample.
==========================================================
"""

from math import ceil
import random

from ..transformation import Transformation
from ....input.component.sample import CorefSample
from ....input.component.field import ListField
__all__ = ['RndRepeat']


class RndRepeat(Transformation):
    r""" 
    Randomly choose trans_p * num_sentences sentences,
    and each of them will be repeated
        somewhere else in the sample.

    Attributes:
        trans_p: proportion of repeated sentences; default 0.2
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
                ['I', 'conquered'], ['Anna', 'bel', 'wanna', 'sleep'], 
                ['Anna', 'bel', 'is', 'happy']],
            'clusters': [
                [[1, 1], [3, 3], [9, 9]], 
                [[5, 6], [11, 12], [17, 18]]]}

    """

    def __init__(self, trans_p=0.2, **kwargs):
        super().__init__()
        self.trans_p = trans_p

    def __repr__(self):
        return 'RndRepeat'

    def _transform(self, sample, n=5, **kwargs):
        r"""
        :param ~textflint.CorefSample sample: a CorefSample
        :param str|list fields: Not used
        :param int n: optional; number of generated samples
        :return list: samples_tfed, transformed sample list.

        """
        if sample.num_sentences() == 0: return [sample] * n
        num_sentences = sample.num_sentences()
        samples_tfed = []

        for i in range(n):
            sample_tfed = CorefSample(sample.dump())
            # repeat times: trans_p * num_sentences; at least 1
            for j in range(ceil(num_sentences * self.trans_p)):
                # randomly choose the sentence to repeat
                ori_sen_idx = int(random.random() * (num_sentences))
                # s_pt = sample.part_conll([ori_sen_idx])
                k_sen = sample.get_kth_sen(ori_sen_idx)
                clusters_pt = sample.part_conll([ori_sen_idx])\
                    .clusters.field_value

                # randomly choose tfed_sen_idx:
                # k_sen will be inserted after position tfed_sen_idx
                # tfed_sen_idx in [0, num_sentences + j - 1):
                # tfed_sen_idx cannot be the last one
                assert sample_tfed.num_sentences() == num_sentences + j, \
                    "Assert failed in RndRepeat: " \
                    "document length does not match."
                tfed_sen_idx = int(random.random() * (num_sentences + j - 1))
                sen_map = sample_tfed.sen_map
                insert_at_idx = sum(sen_map[:tfed_sen_idx+1])
                sample_tfed = sample_tfed.insert_field_after_indices(
                    'x', [insert_at_idx-1], [k_sen])

                assert sen_map[tfed_sen_idx] + len(k_sen) \
                    == sample_tfed.sen_map[tfed_sen_idx], \
                    "Assert failed in RndRepeat: sentence lengths " \
                    "seem to be unexpected after insert. " \
                    "Original sen_map: {0}, current sen_map: {1}"\
                        .format(sen_map, sample_tfed.sen_map)
                sample_tfed.sen_map = sample_tfed.sen_map[:tfed_sen_idx] \
                    + [sen_map[tfed_sen_idx], len(k_sen)] \
                    + sample_tfed.sen_map[tfed_sen_idx+1:]

                setattr(sample_tfed, 'clusters', 
                    ListField([c1 + c2 for c1, c2 in zip(
                        [
                            [[b+insert_at_idx, e+insert_at_idx] for [b, e] in c] 
                            for c in clusters_pt], 
                        sample_tfed.clusters.field_value)]))

            # get the tfed sample and append to list
            samples_tfed.append(sample_tfed)
        return samples_tfed

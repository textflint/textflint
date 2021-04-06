r"""
Add adverb word before verb word with given pos tags
==========================================================
"""

__all__ = ["InsertAdv"]

import random

from ....common.settings import ADVERB_PATH
from ...transformation import Transformation
from ....common.utils.load import plain_lines_loader
from ....common.utils.list_op import trade_off_sub_words
from ....common.utils.install import download_if_needed


class InsertAdv(Transformation):
    r"""
    Transforms an input by add adverb word before verb.

    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.adverb_list = plain_lines_loader(download_if_needed(ADVERB_PATH))

    def __repr__(self):
        return 'InsertAdv'

    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform text string according transform_field.

        :param ~Sample sample: input data, normally one data component.
        :param str field: indicate which field to transform
        :param int n: number of generated samples
        :param kwargs:
        :return list trans_samples: transformed sample list.

        """
        pos_tags = sample.get_pos(field)
        _insert_indices = self._get_verb_location(pos_tags)

        if not _insert_indices:
            return []

        insert_words = []
        insert_indices = []

        for index in _insert_indices:
            _insert_words = self._get_random_adverbs(n=n)

            if _insert_words:
                insert_indices.append(index)
                insert_words.append(_insert_words)

        if not insert_words:
            return []

        insert_words, insert_indices = trade_off_sub_words(
            insert_words, insert_indices, n=n)
        trans_samples = []

        # get substitute candidates combinations
        for i in range(len(insert_words)):
            single_insert_words = insert_words[i]
            trans_samples.append(
                sample.insert_field_before_indices(
                    field, insert_indices, single_insert_words))

        return trans_samples

    @staticmethod
    def _get_verb_location(pos_tags):
        verb_location = []

        for i, pos in enumerate(pos_tags):
            if pos in ['VB', 'VBP', 'VBZ', 'VBG', 'VBD', 'VBN']:
                verb_location.append(i)

        return verb_location

    def _get_random_adverbs(self, n):
        sample_num = min(n, len(self.adverb_list))

        return random.sample(self.adverb_list, sample_num)


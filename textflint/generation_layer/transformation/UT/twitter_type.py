r"""
contract sentence by common abbreviations in TwitterType
==========================================================
"""

__all__ = ['TwitterType']

import random
import string
import numpy as np

from ..transformation import Transformation
from ....common.settings import TWITTER_PATH
from ....common.utils.load import json_loader
from ....common.utils.install import download_if_needed


class TwitterType(Transformation):
    r"""
    Transforms input by common abbreviations in TwitterType.

    :param str mode: Twitter type, only support ['at', 'url', 'random']

    """
    def __init__(
        self,
        mode='random',
        **kwargs
    ):
        super().__init__()
        self.mode = mode
        if mode and mode not in ['at', 'url', 'random']:
            raise ValueError(f"{mode} not one of ['at', 'url', 'random']")
        self.twitter_dic = json_loader(download_if_needed(TWITTER_PATH))

    def __repr__(self):
        return 'TwitterType' + '_' + self.mode

    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform text string according transform_field.

        :param ~Sample sample: input data, normally one data component.
        :param str|list field: indicate which field to transform.
        :param int n: number of generated samples
        :param kwargs:
        :return list trans_samples: transformed sample list.

        """
        trans_samples = []
        # replace sub strings by contractions
        indices, contractions = self._get_contractions(sample, field)

        if indices:
            contract_sample = sample.unequal_replace_field_at_indices(
                field, indices, contractions)
        else:
            contract_sample = sample

        random_texts = self._get_random_text(n=n)

        for random_text in random_texts:
            insert_beginning = random.choice([True, False])
            # insert at the beginning
            if insert_beginning:
                trans_sample = contract_sample.insert_field_before_index(
                    field, 0, random_text)
            else:
                end_index = len(contract_sample.get_words(field)) - 1
                trans_sample = contract_sample.insert_field_after_index(
                    field, end_index, random_text)
            trans_samples.append(trans_sample)

        return trans_samples

    def _get_contractions(self, sample, field):
        r"""
        :param Sample sample: Sample
        :param str field: field str
        :return list indices: list of contractions indices list
        :return list contractions: list of contractions list

        """
        tokens = sample.get_words(field)
        contractions = []
        indices = []

        for twitter_phrase in self.twitter_dic:
            twitter_words = self.processor.tokenize(twitter_phrase)

            for i in range(len(tokens) - len(twitter_words)):
                if tokens[i: len(twitter_words)] == twitter_words:
                    contractions.append(self.twitter_dic[twitter_phrase])
                    indices.append([i, i + len(twitter_words)])

        return indices, contractions

    def _get_random_text(self, n=1):
        random_texts = []
        for i in range(n):
            mode = self.mode if self.mode != 'random' \
                else random.choice(['at', 'url'])
            if mode == 'at':
                random_text = self.random_at(random.randint(1, 10))
            else:
                random_text = self.random_url(random.randint(1, 5))
            random_texts.append(random_text)

        return random_texts

    @staticmethod
    def random_string(n):
        return ''.join(np.random.choice(
            [x for x in string.ascii_letters + string.digits], n))

    def random_url(self, n=5):
        return 'https://{0}.{1}/{2}'.format(
            self.random_string(n),
            self.random_string(n),
            self.random_string(n))

    def random_at(self, n=5):
        return '@{0}'.format(self.random_string(n))


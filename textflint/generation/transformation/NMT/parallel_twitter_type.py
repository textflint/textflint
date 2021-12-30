r"""
parallelly contract sentence by common abbreviations in TwitterType
==========================================================
"""

__all__ = ['ParallelTwitterType']

import random
import string
import numpy as np

from ....common.utils.error import FlintError
from ....common.utils import logger
from ..UT.twitter_type import TwitterType


class ParallelTwitterType(TwitterType):
    r"""
    Parallelly transforms input by common abbreviations in TwitterType.

    :param str mode: Twitter type, only support ['at', 'url', 'random']

    """
    def __init__(
        self,
        mode='random',
        **kwargs
    ):
        super().__init__(mode)

    def __repr__(self):
        return 'ParallelTwitterType' + '_' + self.mode

    def _transform(self, sample, field=['source', 'target'], n=1, **kwargs):
        r"""
        Transform text string according transform_field.

        :param ~Sample sample: input data, normally one data component.
        :param str|list field: indicate which field to transform.
        :param int n: number of generated samples
        :param kwargs:
        :return list trans_samples: transformed sample list.

        """
        trans_samples = []
        contract_sample = sample

        random_texts = self._get_random_text(n=n)

        for random_text in random_texts:
            insert_beginning = random.choice([True, False])
            # insert at the beginning
            if insert_beginning:
                trans_sample = contract_sample.insert_field_before_index(
                    field[0], 0, random_text).insert_field_before_index(
                    field[1], 0, random_text)
            else:
                source_end_index = len(contract_sample.get_words(field[0])) - 1
                target_end_index = len(contract_sample.get_words(field[1])) - 1
                trans_sample = contract_sample.insert_field_after_index(
                    field[0], source_end_index, random_text).insert_field_after_index(
                    field[1], target_end_index, random_text)
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

        source_tokens = sample.get_words(field[0])
        target_tokens = sample.get_words(field[1])

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

    def transform(self, sample, n=1, field=['source', 'target'], split_by_space=False, **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~textflint.input.component.sample.Sample sample: Data
            sample for augmentation.
        :param int n: Max number of unique augmented output, default is 5.
        :param str|list field: Indicate which fields to apply transformations.
        :param dict **kwargs: other auxiliary params.
        :return: list of Sample

        """
        if n < 1:
            return []

        if not isinstance(field, list):
            assert isinstance(field, str), "The type of field must be a str " \
                                           "or list not {0}".format(type(field))
            fields = [field]
        else:
            fields = field

        assert isinstance(fields, list), \
            "The type of field can choice in str or" \
            " list,not {0}".format(type(field))
        fields = list(set(fields))

        try:  # Deal with textflint Exception
            if len(fields) != 2:
                raise ValueError("The length of fields must be 2, not {0}.".format(len(fields)))
            else:
                if not ((fields[0] == 'source' and fields[1] == 'target') or 
                        (fields[0] == 'target' and fields[1] == 'source')):
                    raise ValueError("The elements of fields must be 'source' and 'target', not {0}.".format(fields))
                fields.sort()
                transform_results = self._transform(sample, n=n,
                                                    field=fields, split_by_space=split_by_space, **kwargs)
        except FlintError as e:
            logger.error(str(e))
            return []
        except Exception as e:
            logger.error(str(e))
            raise FlintError("You hit an internal error. "
                             "Please open an issue in "
                             "https://github.com/textflint/textflint"
                             " to report it.")
        if transform_results:
            return [sample for sample in transform_results
                    if (not sample.is_origin) and sample.is_legal()]
        else:
            return []
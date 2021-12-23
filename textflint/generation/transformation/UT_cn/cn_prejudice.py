r"""
Reverse gender or place names in sentences
==========================================================
"""

__all__ = ['CnPrejudice']

import random

from ..transformation import Transformation
from ....common.utils.file_io import read_json
from ....common.utils.list_op import descartes
from ....common.utils.load import plain_lines_loader
from ....common.utils.install import download_if_needed
from ....common.settings import PREJUDICE_PATH, PREJUDICE_WORD_PATH, \
    PREJUDICE_LOC_PATH, PREJUDICE_LOC2IDX


class CnPrejudice(Transformation):
    r"""
    Transforms an input by Reverse gender or place names in sentences.

    """
    def __init__(
        self,
        change_type='Loc',
        prejudice_tendency=None,
        **kwargs
    ):
        r"""
        :param str change_type: change type, only support ['Name', 'Loc']
        :param str prejudice_tendency: prejudice tendency, choose different
            tendency according to change_type

        """
        super().__init__()
        if change_type == 'Name':
            self.flag_type = True
            if not prejudice_tendency or prejudice_tendency == 'woman':
                self.prejudice_tendency = 'woman'
            elif prejudice_tendency == 'man':
                self.prejudice_tendency = 'man'
            else:
                raise ValueError(
                    'Prejudice tendency not support name type {0}, '
                    'please choose change type from woman and man'.
                    format(change_type))
            self.type = self.prejudice_tendency
            # self.man_name, self.woman_name = self.get_data(
            #     download_if_needed(PREJUDICE_PATH))
            # self.word = self.get_word(
            #     download_if_needed(PREJUDICE_WORD_PATH),
            #     prejudice_tendency)
            self.man_name = ['小明', '小强', '小刚']
            self.woman_name = ['小丽', '小娜']
            self.word = {
                '他': '她',
                '他的': '她的',
            }
        elif change_type == 'Loc':
            self.flag_type = False
            if not prejudice_tendency:
                prejudice_tendency = ['America']
            elif isinstance(prejudice_tendency, str):
                prejudice_tendency = [prejudice_tendency]
            assert isinstance(prejudice_tendency, list)
            for prejudice_type in prejudice_tendency:
                if prejudice_type not in PREJUDICE_LOC2IDX:
                    raise ValueError(
                        'Prejudice tendency not support name type {0}, please '
                        'choose change type from America, Europe,Africa,China,'
                        'Japan,India'.format(prejudice_type))
            self.prejudice_tendency = [PREJUDICE_LOC2IDX[i] for i in prejudice_tendency]
            self.max_len_loc = 0
            self.loc2idx = {'纽约': 1, '华盛顿': 1, '巴黎': 2, '柏林': 2}
            self.idx2loc = {
                1: ['纽约', '华盛顿'],
                2: ['巴黎', '柏林'],
            }
            self.type = ''.join([k[:2] for k in prejudice_tendency])
            self.prejudice_data = [
                j for i in self.prejudice_tendency for j in self.idx2loc[i]]

        else:
            raise ValueError(
                'Prejudice not support name type {0}, please choose'
                ' change type from Name and Loc'.format(change_type))

    def __repr__(self):
        return 'CnPrejudice' + '-' + \
               ['Name', 'Loc'][1 - self.flag_type] + '-' + self.type

    def _transform(self, sample, field='x', n=5, **kwargs):
        r"""
        Transform text string according transform_field.

        :param ~Sample sample: input data, normally one data component.
        :param str field:  indicate which field to transform.
        :param int n: number of generated samples
        :param kwargs:
        :return list trans_samples: transformed sample list.

        """
        # get the word in sentence
        words = sample.get_words(field)
        words_indices = []
        idx = 0
        for word in words:
            words_indices.append((idx, idx+len(word)))
            idx += len(word)

        trans_samples = []
        if self.flag_type:
            change_pos, change_item = self._get_name_change(words, words_indices, n)
        else:
            change_pos, change_item = self._get_loc_change(words, words_indices, n)
        if change_pos:
            change_item = descartes(change_item, n)

        n = min(n, len(change_item))

        for i in range(n):
            # get the change position and change items
            trans_sample = sample.unequal_replace_field_at_indices(
                field, change_pos, change_item[i])
            trans_samples.append(trans_sample)

        return trans_samples

    def _get_name_change(self, words, words_indices, n):
        r"""
        Find the location and name to replace.

        :param list words: input sentence's tokens.
        :param int n: number of generated samples
        :return list change_pos: transformed pos list.
        :return list change_items: transformed items list.

        """
        # Find the location of the name
        change_pos = []
        change_items = []
        flag = 0

        for i in range(len(words)):
            if not flag:
                if words[i] in self.woman_name \
                        and self.prejudice_tendency == 'man':
                    change_pos.append(words_indices[i])
                    change_items.append(random.sample(self.man_name, n))
                    flag = 1
                elif words[i] in self.man_name \
                        and self.prejudice_tendency == 'woman':
                    change_pos.append(words_indices[i])
                    change_items.append(random.sample(self.woman_name, n))
                    flag = 1
            else:
                flag = 0
            if not flag and words[i] in self.word:
                change_pos.append(words_indices[i])
                change_items.append([self.word[words[i]]])

        return change_pos, change_items

    def _get_loc_change(self, words, words_indices, n):
        r"""
        Find the location and name to replace.

        :param list words: input sentence's tokens.
        :param int n: number of generated samples
        :return list change_pos: transformed pos list.
        :return list change_items: transformed items list.

        """
        change_pos = []
        change_items = []

        for i in range(len(words)):
            if words[i] in self.loc2idx and self.loc2idx[words[i]] not in self.prejudice_tendency:
                change_pos.append(words_indices[i])
                change_items.append(random.sample(self.prejudice_data, n))
        return change_pos, change_items


r"""
Reverse gender or place names in sentences
==========================================================
"""

__all__ = ['Prejudice']

import random

from ..transformation import Transformation
from ....common.utils.file_io import read_json
from ....common.utils.list_op import descartes
from ....common.utils.load import plain_lines_loader
from ....common.utils.install import download_if_needed
from ....common.settings import PREJUDICE_PATH, PREJUDICE_WORD_PATH, \
    PREJUDICE_LOC_PATH, PREJUDICE_LOC2IDX


class Prejudice(Transformation):
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
            self.man_name, self.woman_name = self.get_data(
                download_if_needed(PREJUDICE_PATH))
            self.word = self.get_word(
                download_if_needed(PREJUDICE_WORD_PATH),
                prejudice_tendency)
        elif change_type == 'Loc':
            self.flag_type = False
            if not prejudice_tendency:
                prejudice_tendency = ['Africa']
            elif isinstance(prejudice_tendency, str):
                prejudice_tendency = [prejudice_tendency]
            assert isinstance(prejudice_tendency, list)
            for prejudice_type in prejudice_tendency:
                if prejudice_type not in PREJUDICE_LOC2IDX:
                    raise ValueError(
                        'Prejudice tendency not support name type {0}, please '
                        'choose change type from America, Europe,Africa,China,'
                        'Japan,India'.format(prejudice_type))
            self.prejudice_tendency = [PREJUDICE_LOC2IDX[i]
                                       for i in prejudice_tendency]
            self.max_len_loc = 0
            self.loc2idx, self.idx2loc = self.get_loc_data(
                download_if_needed(PREJUDICE_LOC_PATH))
            self.type = ''.join([k[:2] for k in prejudice_tendency])
            self.prejudice_data = [
                j for i in self.prejudice_tendency for j in self.idx2loc[i]]

        else:
            raise ValueError(
                'Prejudice not support name type {0}, please choose'
                ' change type from Name and Loc'.format(change_type))

    def __repr__(self):
        return 'Prejudice' + '-' + \
               ['Name', 'Loc'][1 - self.flag_type] + '-' + self.type

    @staticmethod
    def get_data(path):
        # get the name dictionary
        for dic in read_json(path):
            _, dic = dic
            return dic['men'], dic['women']

    def get_loc_data(self, path):
        # Get some regions and place names
        loc2idx = {}
        idx2loc = {}
        m = 0
        for line in plain_lines_loader(path):
            line = line.strip().split('\t')
            line[1] = int(line[1])
            loc2idx[line[0]] = line[1]
            m = max(m, len(line[0].split(' ')))
            if line[1] in idx2loc:
                idx2loc[line[1]].append(line[0])
            else:
                idx2loc[line[1]] = [line[0]]
        self.max_len_loc = m
        return loc2idx, idx2loc

    @staticmethod
    def get_word(path, prejudice_tendency):
        res = {}
        # Get the personal pronouns that need to be replaced
        flag = 1 if prejudice_tendency == 'man' else 0
        for line in plain_lines_loader(path):
            line = line.split(' ')
            res[line[flag]] = line[1 - flag]
            res[line[flag].upper()] = line[1 - flag].upper()
            res[line[flag].title()] = line[1 - flag].title()
        return res

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
        trans_samples = []
        if self.flag_type:
            change_pos, change_item = self._get_name_change(words, n)
        else:
            change_pos, change_item = self._get_loc_change(words, n)

        if change_pos:
            change_item = descartes(change_item, n)

        n = min(n, len(change_item))

        for i in range(n):
            # get the change position and change items
            trans_sample = sample.unequal_replace_field_at_indices(
                field, change_pos, change_item[i])
            trans_samples.append(trans_sample)

        return trans_samples

    def _get_name_change(self, words, n):
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
                    change_pos.append(i)
                    change_items.append(random.sample(self.man_name, n))
                    flag = 1
                elif words[i] in self.man_name \
                        and self.prejudice_tendency == 'woman':
                    change_pos.append(i)
                    change_items.append(random.sample(self.woman_name, n))
                    flag = 1
            else:
                flag = 0
            if not flag and words[i] in self.word:
                change_pos.append(i)
                change_items.append([self.word[words[i]]])

        return change_pos, change_items

    def _get_loc_change(self, words, n):
        r"""
        Find the location and name to replace.

        :param list words: input sentence's tokens.
        :param int n: number of generated samples
        :return list change_pos: transformed pos list.
        :return list change_items: transformed items list.

        """
        change_pos = []
        change_items = []
        start = 0
        # get the pos of change and change items
        while start < len(words):
            end = start + 1
            while end - start <= self.max_len_loc and end < len(words):
                word = self.processor.inverse_tokenize(words[start:end])
                if word in self.loc2idx \
                        and self.loc2idx[word] not in self.prejudice_tendency:
                    change_pos.append([start, end])
                    change_items.append(random.sample(self.prejudice_data, n))
                    break
                end += 1
            start = end
        return change_pos, change_items


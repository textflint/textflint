r"""
Replace word with its SwapSyn.
==========================================================
"""
__all__ = ["SwapSyn"]
from ..transformation import Transformation
from ....common.settings import SYNONYM_PATH
from ....common.utils.load import plain_lines_loader
from ....common.utils.install import download_if_needed
from ....common.utils.list_op import descartes
import random


class SwapSyn(Transformation):
    r"""
    Replace word with its synonym.

    Example::

        先生过奖了 -> 先生过誉了

    """

    def __init__(self, **kwargs):
        r"""
        :param dict synonym_dict: the dictionary of synonym

        """
        super().__init__()
        self.synonym_dict = self.make_dict(download_if_needed(SYNONYM_PATH))

    def __repr__(self):
        return 'SwapSyn'

    @staticmethod
    def make_dict(path):
        r"""
        read data and make dictionary

        :param str path: the path of data
        :return dict: the dict of data

        """
        dic = {}
        lines = plain_lines_loader(path)
        for line in lines:
            line = line.strip().split(' ')
            if line[0] not in dic:
                dic[line[0]] = []
            for word in line[1:]:
                if word not in dic[line[0]]:
                    dic[line[0]].append(word)
        return dic

    def _transform(self, sample, n=5, **kwargs):
        r"""
        In this function, there are several deformation modes.

        :param ~textflint.CWSSample sample: the data which need be changed
        :param **kwargs:
        :return: In this function, there may be multiple outputs

        """
        # get sentence words
        origin_words = sample.get_words()

        # change function
        change_pos, change_word = self._get_transformations(origin_words, n)
        if len(change_pos) == 0:
            return []

        change_word = descartes(change_word, n)
        return [sample.replace_at_ranges(change_pos, words) for words in change_word]

    def _get_transformations(self, words, n):
        r"""
        Replace synonym function

        :param list words: chinese sentence words
        :param int n: the number of transformations
        :return list: two list include the pos which changed the word which
            changed and the label which changed

        """
        start = 0
        change_pos = []
        change_word = []

        for word in words:
            # find the word
            if word in self.synonym_dict:
                # save synonyms and change word segmentation labels
                change_pos.append([start, start + len(word)])
                change_word += [random.sample(
                    self.synonym_dict[word],
                    min(n, len(self.synonym_dict[word])))]
            start += len(word)

        return change_pos, change_word

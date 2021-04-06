r"""
Make the first word of the surname and the preceding word form a word,
            and the last word of the name and the following word form a word
==========================================================
"""
__all__ = ["SwapName"]
import random

from ..transformation import Transformation
from ....common.settings import NAME_PATH, WORD_LIST_PATH
from ....common.utils.load import plain_lines_loader
from ....common.utils.install import download_if_needed
from ....common.utils.list_op import descartes


class SwapName(Transformation):
    r"""
    Make the first word of the surname and the preceding word form a word,
            and the last word of the name and the following word form a word

    Example::

        我朝小明走了过去 -> 我朝向明走了过去

    """

    def __init__(self, **kwargs):
        r"""
        :param list firstname_list: family name dictionary
        :param list word_list: dictionary of words
        :param dict word_end_dict: a dictionary
        :param dict name_dict: A dictionary ending with a surname

        """
        super().__init__()
        self.firstname_list = plain_lines_loader(download_if_needed(NAME_PATH))
        self.word_list = plain_lines_loader(download_if_needed(WORD_LIST_PATH))
        self.word_end_dict, self.name_dict = self.make_dict()

    def __repr__(self):
        return 'SwapName'

    def make_dict(self):
        r"""
        :return: Last name dictionary and first name dictionary

        """
        word_end_dict = {}
        name_dict = {}

        for word in self.word_list:
            if len(word) > 1:
                if word[1:] not in word_end_dict:
                    word_end_dict[word[1:]] = [word[0]]
                elif word[0] not in word_end_dict[word[1:]]:
                    word_end_dict[word[1:]] += [word[0]]

                if word[-1:] in self.firstname_list:
                    if word[:-1] not in name_dict:
                        name_dict[word[:-1]] = [word[-1:]]
                    elif word[-1:] not in name_dict[word[:-1]]:
                        name_dict[word[:-1]].append(word[-1])

        return word_end_dict, name_dict

    def _transform(self, sample, n=5, **kwargs):
        r"""
        We randomly generated five sets of data.

        :param ~textflint.CWSSample sample: sample the data which need be changed
        :param int n: number of generated data
        :param **kwargs:
        :return: trans_sample a list of sample
        """
        # get sentence and label and ner_label
        origin_sentence = sample.get_value('x')
        origin_label = sample.get_value('y')
        ner_label, _ = sample.ner

        # change function
        change_pos, change_list = self._get_transformations(
            origin_sentence, origin_label, ner_label, n)

        if len(change_pos) == 0:
            return []

        change_list = descartes(change_list, n)
        return [sample.replace_at_ranges(change_pos, item)
                for item in change_list]

    def _get_transformations(self, sentence, label, ner_label, n):
        r"""
        transformation function

        :param str sentence: chinese sentence
        :param list label: Chinese word segmentation tag
        :param list ner_label: sentence's ner tag
        :param int n: the number of transformations
        :return list: two list include the pos which changed and the
            label which changed
        """
        assert len(sentence) == len(label)

        change_pos = []
        change_list = []
        if len(ner_label):
            for ner in ner_label:
                tag, start, end = ner
                # Determine whether it is a name based on the ner tag
                # and the word segmentation tag
                if tag != 'Nh' or label[start] != 'B' \
                        or label[end] != 'E' \
                        or label[start + 1:end] != ['M'] * (end - start - 1):
                    continue
                # Combine the last name and the previous n words into a word,
                # and get a list of replacement words
                s = ''
                change = []

                for i in range(1, 6):
                    if start < i:
                        break
                    s = sentence[start - i] + s
                    if s in self.name_dict:
                        change += self.name_dict[s]
                if len(change) > 0:
                    change_pos += [start]
                    change_list += [random.sample(change, min(len(change), n))]

                # The name and the following n letters form a word,
                # and get a list of replacement words
                s = ''
                change = []

                for j in range(1, 5):
                    if end + j >= len(label):
                        break
                    s += sentence[end + j]
                    if s in self.word_end_dict:
                        change += self.word_end_dict[s]
                if len(change) > 0:
                    change_pos += [end]
                    change_list += [random.sample(change, min(len(change), n))]

        return change_pos, change_list

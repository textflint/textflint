r"""
Replace verb with detachable words or a one a.
==========================================================
"""
__all__ = ["SwapVerb"]
from ..transformation import Transformation
from ....common.settings import DETACHABLE_WORD_PATH, AONEA_PATH
from ....common.utils.load import plain_lines_loader
from ....common.utils.install import download_if_needed
from ....common.preprocess.cn_processor import CnProcessor
import random


class SwapVerb(Transformation):
    r"""
    Replace verb with detachable words or a one a.

    """

    def __init__(self, **kwargs):
        r"""
        :param list detachable_word_list: word can be replaced
            detachable word dictionary
        :param list AoneA_list: word can be replaced by a one a dictionary

        """
        super().__init__()
        self.AoneA_list = plain_lines_loader(download_if_needed(AONEA_PATH))
        self.detachable_word_list = plain_lines_loader(
            download_if_needed(DETACHABLE_WORD_PATH))

    def __repr__(self):
        return 'SwapVerb'

    @staticmethod
    def get_pos_tag_list(pos_tag):
        r"""
        get the list of pos tag

        :param list pos_tag: transform tuple to list

        """
        # get the list of pos tag
        pos_list = []
        for i in pos_tag:
            tag, start, end = i
            pos_list += [tag] * (end - start + 1)
        return pos_list

    def _transform(self, sample, n=5, **kwargs):
        r"""
        In this function, because there is only one deformation mode,
            only one set of outputs is output.

        :param ~textflint.CWSSample sample: the data which need be changed
        :param int n: the number of transformations
        :param **kwargs:
        :return: In this function, because there is only one deformation mode,
            only one set of outputs is output.

        """
        # get sentence token
        origin_words = sample.get_words()
        origin_labels = sample.get_value('y')
        pos_tags = sample.pos_tags
        pos_tags = self.get_pos_tag_list(pos_tags)
        trans_sample = []
        # change function
        change_pos, change_sentence, change_label = \
            self._get_transformations(origin_words, pos_tags, origin_labels)
        if len(change_pos) == 0:
            return []
        change_sample = sample.replace_at_ranges(
            change_pos, change_sentence, change_label)
        trans_sample.append(change_sample)

        return trans_sample

    def _get_transformations(self, words, pos_tags, labels):
        r"""
        Reduplication change function.

        :param list words: chinese sentence words
        :param list pos_tags: chinese sentence pos tags
        :param list labels: chinese sentence segmentation label
        :return: three list include the pos which changed the word which
            changed and the label which changed

        """

        change_pos = []
        change_sentence = []
        change_label = []
        start = 0

        for word in words:
            # get every the word
            # find the word in AoneA dictionary
            flag1 = False
            flag2 = False
            if start + 1 < len(labels) and pos_tags[start] == 'v' \
                    and self.check_part_pos(word[1:]) \
                    and word[0] in self.AoneA_list:
                flag1 = True
            if word in self.detachable_word_list:
                flag2 = True
            # choice one method to change sentence
            if flag1 and flag2:
                t = random.choice([0, 1])
                if t == 0:
                    flag2 = False
                else:
                    flag1 = False
            if flag1:
                change_pos.append([start, start + 1])
                change_sentence.append(word[0] + '一'
                                       + word[0])
                if len(word) == 1:
                    change_label.append(['B', 'M', 'E'])
                else:
                    change_label.append(['B', 'M', 'M'])
            elif flag2:
                change_pos.append([start, start + len(word)])
                change_sentence.append(word[0] + '了个'
                                       + word[1:])
                change_label.append(['S'] * 4)

            start += len(word)

        return change_pos, change_sentence, change_label

    @staticmethod
    def check_part_pos(sentence):
        """
        get the pos of sentence if we need

        :param str sentence: origin word
        :return: bool
        """
        if sentence == "":
            return False
        Processor = CnProcessor()
        pos = Processor.get_pos_tag(sentence)
        if len(pos) == 1 and pos[0][0] == 'n':
            return True
        return False

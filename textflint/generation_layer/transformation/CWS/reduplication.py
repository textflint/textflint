r"""
Replace word with reduplication such as AABB or ABAB.
==========================================================
"""
__all__ = ["Reduplication"]
from ..transformation import Transformation
from ....common.settings import AABB_PATH, ABAB_PATH
from ....common.utils.load import plain_lines_loader
from ....common.utils.install import download_if_needed


class Reduplication(Transformation):
    r"""
    Replace word with reduplication such as AABB or ABAB.

    Example::

        朦胧的月色 -> 朦朦胧胧的月色

    """

    def __init__(self, **kwargs):
        r"""
        :param list ABAB_list: word can be replaced by abab dictionary
        :param list AABB_list: word can be replaced by aabb dictionary
        :param **kwargs:

        """
        super().__init__()
        self.ABAB_list = plain_lines_loader(download_if_needed(ABAB_PATH))
        self.AABB_list = plain_lines_loader(download_if_needed(AABB_PATH))

    def __repr__(self):
        return 'Reduplication'

    def _transform(self, sample, n=5, **kwargs):
        r"""
        In this function, because there is only one deformation mode,
            only one set of outputs is output.

        :param ~textflint.CWSSample: the data which need be changed
        :param **kwargs:
        :return: In this function, because there is only one deformation mode,
            only one set of outputs is output

        """
        # get sentence token
        origin_words = sample.get_words()
        # change function
        change_pos, change_sentence, change_label = \
            self._get_transformations(origin_words)
        if len(change_pos) == 0:
            return []
        change_sample = sample.replace_at_ranges(
            change_pos, change_sentence, change_label)

        return [change_sample]

    def _get_transformations(self, words):
        r"""
        Reduplication change function.

        :param list words: chinese sentence words
        :return list: three list include the pos which changed the word which
            changed and the label which changed

        """

        change_pos = []
        change_sentence = []
        change_label = []
        start = 0

        for word in words:
            # get every the word
            # find the word in AABB dictionary
            if len(word) == 2:
                if word in self.AABB_list:
                    # pos_tag[start:start + 2] == ['v', 'v']
                    change_pos.append([start, start + 2])
                    change_sentence.append(
                        word[0] + word[0] + word[1] + word[1])
                    change_label.append(['B', 'M', 'M', 'E'])
                # find ABAB word
                elif word in self.ABAB_list:
                    # pos_tag[start:start + 2] == ['v', 'v']
                    change_pos.append([start, start + 2])
                    change_sentence.append(word + word)
                    change_label.append(['B', 'M', 'M', 'E'])
            start += len(word)

        return change_pos, change_sentence, change_label

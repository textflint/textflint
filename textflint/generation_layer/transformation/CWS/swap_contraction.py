r"""
Replace abbreviations with full names.
==========================================================
"""
__all__ = ["SwapContraction"]
from ..transformation import Transformation
from ....common.settings import abbreviation_path
from ....common.utils.load import plain_lines_loader
from ....common.utils.install import download_if_needed


class SwapContraction(Transformation):
    r"""
    Replace abbreviations with full names.

    Example::

        央视 -> 中央电视台

    """

    def __init__(self, **kwargs):
        r"""
        :param dict abbreviation_dict: the dictionary of abbreviation
        :param **kwargs:

        """
        super().__init__()
        self.abbreviation_dict = self.make_dict(
            download_if_needed(abbreviation_path))

    def __repr__(self):
        return 'SwapContraction'

    @staticmethod
    def make_dict(path):
        r"""
        read data

        :param str path: the path of data
        :return: the dic of data

        """
        dic = {}
        lines = plain_lines_loader(path)
        for line in lines:
            line = line.strip().split(' ')
            dic[line[0]] = line[1:]
        return dic

    def _transform(self, sample, n=1, **kwargs):
        r"""
        Transform the sample.

        :param ~textflint.CWSSample sample: the data which need be changed
        :param **kwargs:
        :return: In this function, because there is only one deformation mode,
            only one set of outputs is output

        """
        # get sentence and label
        origin_words = sample.get_words()
        # change function
        change_pos, change_sentence, change_label = self._get_transformations(
            origin_words)
        if len(change_pos) == 0:
            return []
        change_sample = sample.replace_at_ranges(
            change_pos, change_sentence, change_label)

        return [change_sample]

    def _get_transformations(self, words):
        r"""
        Replace abbreviation function

        :param list words: chinese sentence words
        :return: change_pos, change_sentence, change_label
                three list include the pos which changed the word which changed
                and the label which changed

        """
        assert isinstance(words, list), \
            'The type of wrods must be a list not {0}'.format(type(words))
        start = 0
        change_pos = []
        change_sentence = []
        change_label = []

        for word in words:
            # find the abbreviation
            if word in self.abbreviation_dict:
                # save abbreviations and change word segmentation labels
                change_pos.append([start, start + len(word)])
                change_sentence.append(self.abbreviation_dict[word])
                change = []
                for i in self.abbreviation_dict[word]:
                    if len(i) == 1:
                        change.append('S')
                    else:
                        change += ['B'] + ['M'] * (len(i) - 2) + ['E']
                change_label.append(change)
            start += len(word)
        return change_pos, change_sentence, change_label

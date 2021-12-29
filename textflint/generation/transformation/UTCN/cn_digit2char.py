r"""
convert digit to char
==========================================================
"""

__all__ = ['CnDigit2Char']

import random

from ...transformation import CnWordSubstitute


class CnDigit2Char(CnWordSubstitute):
    r"""
    Transforms an input by replacing its numbers.

    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

    def __repr__(self):
        return 'CnDigit2Char'

    def _get_candidates(self, word, pos=None, n=1):
        r"""
        Returns a list containing all possible words .

        :param str word: token word to transform.
        :param str pos: pos tag
        :param int n: max number of candidates
        :return list candidates: candidates list

        """
        n = max(n, 1)
        candidates = []

        if word.isdigit() and int(word) >=0 and int(word) <= 9999:
            number = int(word)
            candidates.append(self.num_to_char(number))

        return candidates

    def num_to_char(self, num):
        _MAPPING = (
        u'零', u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'十', u'十一', u'十二', u'十三', u'十四', u'十五', u'十六', u'十七',
        u'十八', u'十九')
        _P0 = (u'', u'十', u'百', u'千',)
        _S4 = 10 ** 4
        assert (0 <= num and num < _S4)
        if num < 20:
            return _MAPPING[num]
        else:
            lst = []
            while num >= 10:
                lst.append(num % 10)
                num = num / 10
            lst.append(num)
            c = len(lst)  # 位数
            result = u''

            for idx, val in enumerate(lst):
                val = int(val)
                if val != 0:
                    result += _P0[idx] + _MAPPING[val]
                    if idx < c - 1 and lst[idx + 1] == 0:
                        result += u'零'
            return result[::-1]

    def skip_aug(self, words, words_indices, tokens, mask, **kwargs):
        return self.pre_skip_aug(words, words_indices, tokens, mask)


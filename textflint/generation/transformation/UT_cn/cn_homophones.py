r"""
homophones
==========================================================
"""

__all__ = ['CnHomophones']

import random

from ...transformation import CnWordSubstitute
from pypinyin import lazy_pinyin
from Pinyin2Hanzi import DefaultHmmParams
from Pinyin2Hanzi import viterbi

hmmparams = DefaultHmmParams()

class CnHomophones(CnWordSubstitute):
    r"""
    Transforms an input by replacing its numbers.

    """
    def __init__(
        self,
        get_pos=None,
        **kwargs
    ):
        super().__init__(get_pos=get_pos)

    def __repr__(self):
        return 'CnHomophones'

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
        if pos in ['v', 'n']:
            candidates = self.homophones(word, n)

        return candidates

    def homophones(self, word, n):
        pinyin = lazy_pinyin(word)
        result = viterbi(hmm_params=hmmparams, observations=pinyin, path_num=n + 1)
        ret = []
        for i in result:
            if ''.join(i.path) != word:
                ret.append(''.join(i.path))
        return ret

    def skip_aug(self, words, words_indices, tokens, mask, **kwargs):
        return self.pre_skip_aug(words, words_indices, tokens, mask)


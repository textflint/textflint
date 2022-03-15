r"""
homophones
==========================================================
"""

__all__ = ['CnHomophones']

from pypinyin import lazy_pinyin
from Pinyin2Hanzi import DefaultHmmParams
from Pinyin2Hanzi import viterbi

from ...transformation import CnWordSubstitute


hmmparams = DefaultHmmParams()


class CnHomophones(CnWordSubstitute):
    r"""
    Transforms an input by replacing its numbers.

    """
    def __init__(
        self,
        trans_min=1,
        trans_max=10,
        trans_p=0.3,
        stop_words=None,
        islist=False,
        **kwargs
    ):
        super().__init__(
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words,
        )
        self.get_pos = True
        self.islist = islist

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
        try:
            result = viterbi(hmm_params=hmmparams, observations=pinyin, path_num=n + 1)
        except Exception:
            return []
        ret = []
        for i in result:
            if ''.join(i.path) != word:
                ret.append(''.join(i.path))
        return ret

    def skip_aug(self, words, words_indices, tokens, mask, **kwargs):
        if self.islist:
            return self.pre_skip_aug_list(words, words_indices, tokens, mask)
        else:
            return self.pre_skip_aug(words, words_indices, tokens, mask)

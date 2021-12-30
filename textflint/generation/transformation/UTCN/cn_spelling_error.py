r"""
Transformation that apply spelling error simulation to textual input.
==========================================================

"""

__all__ = ['CnSpellingError']

from ...transformation import CnWordSubstitute
from pypinyin import lazy_pinyin
from Pinyin2Hanzi import DefaultHmmParams
from Pinyin2Hanzi import viterbi
import random

hmmparams = DefaultHmmParams()

class CnSpellingError(CnWordSubstitute):
    r"""
    Transformation that leverage pre-defined spelling mistake dictionary to
    simulate spelling mistake.

    https://arxiv.org/ftp/arxiv/papers/1812/1812.04718.pdf

    """
    def __init__(
            self,
            trans_min=1,
            trans_max=10,
            trans_p=0.3,
            stop_words=None,
            include_reverse=True,
            rules_path=None,
            get_pos=None,
            **kwargs
    ):
        r"""
        :param int trans_min: Minimum number of character will be augmented.
        :param int trans_max: Maximum number of character will be augmented.
            If None is passed, number of augmentation is calculated via
            aup_char_p.If calculated result from aug_p is smaller than aug_max,
            will use calculated result from aup_char_p. Otherwise, using
            aug_max.
        :param float trans_p: Percentage of character (per token) will be
            augmented.
        :param list stop_words: List of words which will be skipped from augment
            operation.
        :param bool include_reverse:  whether build reverse map according to
            spelling error list.
        :param str rules_path: rules_path

        """
        super().__init__(
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words,
            get_pos=get_pos
        )

    def __repr__(self):
        return 'CnSpellingError'

    def _get_candidates(self, word, pos=None, n=1):
        r"""
        Get a list of transformed tokens. Default one word replace one char.

        :param str word: token word to transform.
        :param int n: number of transformed tokens to generate.
        :param kwargs:
        :return: candidate list

        """
        candidates = []
        if pos in ['v', 'n']:
            candidates = self.spell(word)

        return candidates

    def skip_aug(self, words, words_indices, tokens, mask, **kwargs):
        return self.pre_skip_aug(words, words_indices, tokens, mask)

    def spell(self, word):
        spell_error_rule = {
            'ao': 'a',
            'ing': 'in',
            'a': 'e',
            'e': 'a',
        }
        pinyin = lazy_pinyin(word)
        replace_index = random.randint(0, len(pinyin)-1)
        replace_str = pinyin[replace_index]
        replaced_str = None
        for k, v in spell_error_rule.items():
            if k in replace_str:
                replaced_str = replace_str.replace(k, v)
                break
        if replaced_str is None:
            return []
        pinyin[replace_index] = replaced_str
        try:
            result = viterbi(hmm_params=hmmparams, observations=pinyin, path_num=1)
        except Exception:
            return []

        ret = []
        for i in result:
            if ''.join(i.path) != word:
                ret.append(''.join(i.path))
        return ret




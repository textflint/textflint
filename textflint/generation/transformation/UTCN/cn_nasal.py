r"""
Change the front and back nasal sounds of Chinese characters
==========================================================
"""

__all__ = ['CnNasal']

from ...transformation import CnWordSubstitute

from pypinyin import lazy_pinyin
from Pinyin2Hanzi import DefaultHmmParams
from Pinyin2Hanzi import viterbi


hmmparams = DefaultHmmParams()


class CnNasal(CnWordSubstitute):
    r"""
    Transforms an input by replacing its tokens with words of mask language
    predicted.
    To accelerate transformation for long text, input single sentence to
    language model rather than whole text.

    """

    def __init__(
        self,
        trans_min=1,
        trans_max=10,
        trans_p=0.1,
        stop_words=None,
        islist=False,
        **kwargs
    ):
        r"""
        :param str masked_model: masked language model to predicate candidates
        :param str device: indicate utilize cpu or which gpu device to run
            neural network
        :param int accrue_threshold: threshold of Bert results to pick
        :param max_sent_size: max_sent_size
        :param int trans_min: Minimum number of character will be augmented.
        :param int trans_max: Maximum number of character will be augmented.
            If None is passed, number of augmentation is calculated via aup_char_p.
            If calculated result from aug_p is smaller than aug_max, will use
            calculated result from aup_char_p. Otherwise, using aug_max.
        :param float trans_p: Percentage of character (per token) will be
            augmented.
        :param list stop_words: List of words which will be skipped from augment
            operation.

        """
        super().__init__(
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words
        )
        self.islist = islist

    def __repr__(self):
        return 'CNNASAL'

    def _get_candidates(self, word, pos=None, n=5, **kwargs):
        r"""
        Get a list of transformed tokens. Default one word replace one char.

        :param str word: token word to transform.
        :param int n: number of transformed tokens to generate.
        :param kwargs:
        :return: candidate list

        """
        pinyins = lazy_pinyin(word)
        new_pinyins = []

        for pinyin in pinyins:
            if ('eng' in pinyin):
                pinyin = pinyin.replace('eng','en')
            elif ('en' in pinyin) and pinyin != 'en':
                pinyin = pinyin.replace('en','eng')

            if ('ang' in pinyin):
                pinyin = pinyin.replace('ang','an')
            elif ('an' in pinyin):
                pinyin = pinyin.replace('an','ang')

            if ('ing' in pinyin):
                pinyin = pinyin.replace('ing','in')
            elif ('in' in pinyin):
                pinyin = pinyin.replace('in','ing')
            new_pinyins.append(pinyin)
        try:
            result = viterbi(hmm_params=hmmparams, observations=new_pinyins, path_num=n + 1)
        except:
            # TODO  add log
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

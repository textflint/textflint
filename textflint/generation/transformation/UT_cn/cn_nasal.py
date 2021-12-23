r"""
Swapping words by Mask Language Model
==========================================================
"""

__all__ = ['CnNasal']

import random
from copy import copy
from ...transformation import WordSubstitute

from pypinyin import lazy_pinyin
from Pinyin2Hanzi import DefaultHmmParams
from Pinyin2Hanzi import viterbi

hmmparams = DefaultHmmParams()
class CnNasal(WordSubstitute):
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
            trans_p=0.2,
            stop_words=None,
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

    def __repr__(self):
        return 'CNNASAL'


    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform text string according field.

        :param Sample sample: input data, normally one data component.
        :param str field: indicate which field to apply transformation
        :param int n: number of generated samples
        :param kwargs:
        :return list trans_samples: transformed sample list.

        """
        tokens = sample.get_tokens(field)
        tokens_mask = sample.get_mask(field)

        #find legal indices
        legal_indices = self.skip_aug(tokens, tokens_mask)

        if not legal_indices:
            return []

        new_tokens = []
        for index in legal_indices:
            new_token_list  = self._get_candidates(tokens[index],n)
            if new_token_list is not None:
                for new_token in new_token_list:
                    new_tokens.append((new_token,index))

        trans_samples = []

        for i in range(len(new_tokens)):
            if i >= n:
                break
            trans_samples.append(
                sample.unequal_replace_field_at_indices(field, [new_tokens[i][1]], [new_tokens[i][0]]))

        return trans_samples


    def _get_candidates(self, token, n):
        r"""
        Get candidates from MLM model.

        :param torch.tensor batch_tokens_tensor: tokens tensor input
        :param torch.tensor segments_tensors: segment input
        :param list mask_indices: indices to predict candidates
        :param list mask_word_pos_list: pos tags of original target words
        :param int n: candidates number
        :return: list candidates
        """
        pinyin = lazy_pinyin(token)[0]
        if pinyin =='en' or pinyin =='an' or pinyin =='ang':
            return None
        if len(pinyin) >= 2 and (pinyin[-2:] =='in' or pinyin[-2:] =='en' or pinyin[-2:] =='an' ):
            pinyin = pinyin+ 'g'
        elif len(pinyin) >= 3 and (pinyin[-3:] =='ing' or pinyin[-3:] =='eng' or pinyin[-2:] =='ang' ):
            pinyin = pinyin[:-1]
        else:
            return None
        result = viterbi(hmm_params=hmmparams, observations=[pinyin], path_num=n + 1)

        ret = []
        for i in result:
            if ''.join(i.path) != token:
                ret.append(''.join(i.path))
        return ret

    def skip_aug(self, tokens, mask, pos=None):
        return self.pre_skip_aug(tokens, mask)


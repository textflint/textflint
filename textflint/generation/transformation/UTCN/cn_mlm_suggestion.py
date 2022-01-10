r"""
Swapping words by Mask Language Model
==========================================================
"""

__all__ = ['MLMSuggestion']

import random

import torch
from copy import copy
from collections import defaultdict

from ....common import device as default_device
from ...transformation import CnWordSubstitute
from ....common.settings import CN_BERT_MODEL_NAME
from ....common.utils.list_op import trade_off_sub_words


class MLMSuggestion(CnWordSubstitute):
    r"""
    Transforms an input by replacing its tokens with words of mask language
    predicted.
    To accelerate transformation for long text, input single sentence to
    language model rather than whole text.

    """

    def __init__(
            self,
            masked_model=None,
            device=None,
            accrue_threshold=1,
            max_sent_size=100,
            trans_min=1,
            trans_max=10,
            trans_p=0.2,
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
        self.device = self.get_device(device) if device else default_device
        self.max_sent_size = max_sent_size
        self.get_pos = True
        self.accrue_threshold = accrue_threshold
        self.masked_model = masked_model if masked_model else CN_BERT_MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.pos_allowed_token_id = None
        self.islist = islist

    def __repr__(self):
        return 'MLMSuggestion'

    @staticmethod
    def get_device(device):
        r"""
        Get gpu or cpu device.
        :param str device: device string
                           "cpu" means use cpu device.
                           "cuda:0" means use gpu device which index is 0.
        :return: device in torch.
        """
        if "cuda" not in device:
            return torch.device("cpu")
        else:
            return torch.device(device)

    def get_model(self):
        r"""
        Loads masked language model to predict candidates.

        """
        from transformers import BertTokenizer, BertForMaskedLM

        self.tokenizer = BertTokenizer.from_pretrained(
            self.masked_model, do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained(self.masked_model)
        self.model.to(self.device)
        self.model.eval()

    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform text string according field.

        :param dict sample: input data, normally one data component.
        :param str fields: indicate which field to apply transformation
        :param int n: number of generated samples
        :return list: transformed sample list.

        """
        text = sample.get_text(field)
        words = sample.get_words(field)
        words_indices = []
        idx = 0
        for word in words:
            words_indices.append((idx, idx+len(word)))
            idx += len(word)

        tokens = sample.get_tokens(field)
        tokens_mask = sample.get_mask(field)

        # return up to (len(sub_indices) * n) candidates
        pos_info = sample.get_pos(field) if self.get_pos else None
        legal_indices = self.skip_aug(words, words_indices, tokens, tokens_mask, pos=pos_info)

        if not legal_indices:
            return []

        sub_words, sub_indices = self._get_substitute_words(tokens, words, legal_indices, pos=pos_info, n=n,text = text)

        # select property candidates
        trans_num = self.get_trans_cnt(len(words))
        sub_words, sub_indices = trade_off_sub_words(sub_words, sub_indices, trans_num, n)

        if not sub_words:
            return []

        trans_samples = []
        for i in range(len(sub_words)):
            single_sub_words = sub_words[i]
            trans_samples.append(
                sample.unequal_replace_field_at_indices(field, sub_indices, single_sub_words))

        return trans_samples

    def _get_substitute_words(self, tokens, words, legal_indices, pos=None, n=5, text = None):
        r"""
        Returns a list containing all possible words .

        :param list words: all words
        :param list legal_indices: indices which has not been skipped
        :param None|list pos: None or list of pos tags
        :param int n: max candidates for each word to be substituted
        :return list: list of list

        """
        # process each legal words to get maximum transformed samples
        legal_words = [''.join(tokens[start:end]) for (start, end) in legal_indices]
        pos = {(i[1], i[2]+1): i[0] for i in pos} if self.get_pos else None
        legal_words_pos = [pos[index] for index in legal_indices] if self.get_pos else None

        candidates_list = []
        candidates_indices = []

        for index, word in enumerate(legal_words):
            _pos = legal_words_pos[index] if self.get_pos else None
            candidates = self._get_candidates(word, pos=_pos, n=n, word_position = legal_indices[index], text = text)
            # filter no word without candidates
            if candidates:
                candidates_indices.append(legal_indices[index])
                candidates_list.append(candidates)
            if index > n*3:
                break
        return candidates_list, candidates_indices


    def _get_candidates(self,word, pos=None, n=5, word_position = None, text = None):
        r"""
        Get candidates from MLM model.

        :param torch.tensor batch_tokens_tensor: tokens tensor input
        :param torch.tensor segments_tensors: segment input
        :param list mask_indices: indices to predict candidates
        :param list mask_word_pos_list: pos tags of original target words
        :param int n: candidates number
        :return: list candidates
        """
        import torch
        if self.model  is None:
            self.get_model()

        word_length = len(word)
        mask_sentence = '[MASK]'*word_length

        if word_position[1] >510:
            return []

        new_text = text[:word_position[0]]+mask_sentence + text[word_position[1]:]
        new_text_tensor = self.tokenizer(new_text,return_tensors="pt",truncation = True, max_length= 512)


        with torch.no_grad():
            output = self.model(**new_text_tensor)
        output = output.logits[0][word_position[0]+1:word_position[1]+1,:]
        output = torch.argsort(output, descending=True,dim = -1)[:,:n+1].transpose(0,1)
        tokens = [self.tokenizer.decode(word,skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(' ','') for word in output]
        tokens = [token for token in tokens if token != word and token != '']
        return tokens


    def skip_aug(self, words, words_indices, tokens, mask, **kwargs):
        if self.islist:
            return self.pre_skip_aug_list(words, words_indices, tokens, mask)
        else:
            return self.pre_skip_aug(words, words_indices, tokens, mask)

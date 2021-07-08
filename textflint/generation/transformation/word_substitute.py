r"""
WordSubstitute Base Class
============================================
"""
__all__ = ["WordSubstitute"]
import string
from abc import abstractmethod

from ..transformation import Transformation
from ...common.settings import STOP_WORDS, ORIGIN
from ...common.utils.list_op import trade_off_sub_words


class WordSubstitute(Transformation):
    r"""
    Word replace transformation to implement normal word replace functions.

    """

    def __init__(
        self,
        trans_min=1,
        trans_max=10,
        trans_p=0.1,
        stop_words=None,
        **kwargs
    ):
        r"""
        :param int trans_min: Minimum number of word will be augmented.
        :param int trans_max: Maximum number of word will be augmented. If None
            is passed, number of augmentation is
            calculated via aup_char_p. If calculated result from aug_p is
            smaller than aug_max, will use calculated
            result from aup_char_p. Otherwise, using aug_max.
        :param float trans_p: Percentage of word will be augmented.
        :param list stop_words: List of words which will be skipped from
            augment operation.
        :param ~textflint.common.preprocess.EnProcessor processor:
        :param bool get_pos: whether pass pos tag to _get_substitute_words API.

        """
        super().__init__()
        self.trans_min = trans_min
        self.trans_max = trans_max
        self.trans_p = trans_p
        self.stop_words = STOP_WORDS if not stop_words else stop_words
        # set this value to avoid meaningless pos tagging
        self.get_pos = False

    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform text string according field.

        :param dict sample: input data, normally one data component.
        :param str fields: indicate which field to apply transformation
        :param int n: number of generated samples
        :return list: transformed sample list.

        """

        tokens = sample.get_words(field)
        tokens_mask = sample.get_mask(field)

        # return up to (len(sub_indices) * n) candidates
        pos_info = sample.get_pos(field) if self.get_pos else None
        legal_indices = self.skip_aug(tokens, tokens_mask, pos=pos_info)

        if not legal_indices:
            return []

        sub_words, sub_indices = self._get_substitute_words(
            tokens, legal_indices, pos=pos_info, n=n)
        # select property candidates
        trans_num = self.get_trans_cnt(len(tokens))
        sub_words, sub_indices = trade_off_sub_words(
            sub_words, sub_indices, trans_num, n)

        if not sub_words:
            return []

        trans_samples = []

        for i in range(len(sub_words)):
            single_sub_words = sub_words[i]
            trans_samples.append(
                sample.replace_field_at_indices(
                    field, sub_indices, single_sub_words))

        return trans_samples

    def _get_substitute_words(self, words, legal_indices, pos=None, n=5):
        r"""
        Returns a list containing all possible words .

        :param list words: all words
        :param list legal_indices: indices which has not been skipped
        :param None|list pos: None or list of pos tags
        :param int n: max candidates for each word to be substituted
        :return list: list of list

        """
        # process each legal words to get maximum transformed samples
        legal_words = [words[index] for index in legal_indices]
        legal_words_pos = [pos[index]
                           for index in legal_indices] if self.get_pos else None

        candidates_list = []
        candidates_indices = []

        for index, word in enumerate(legal_words):
            _pos = legal_words_pos[index] if self.get_pos else None
            candidates = self._get_candidates(word, pos=_pos, n=n)
            # filter no word without candidates
            if candidates:
                candidates_indices.append(legal_indices[index])
                candidates_list.append(
                    self._get_candidates(
                        word, pos=_pos, n=n))

        return candidates_list, candidates_indices

    @abstractmethod
    def _get_candidates(self, word, pos=None, n=5, **kwargs):
        r"""
        Returns a list containing all possible words .

        :param str word:
        :param str pos: the pos tag
        :return list: candidates list

        """
        raise NotImplementedError

    @abstractmethod
    def skip_aug(self, tokens, mask, pos=None):
        r"""
        Returns the index of the replaced tokens.

        :param list tokens: tokenized words or word with pos tag pairs
        :return list: the index of the replaced tokens

        """
        raise NotImplementedError

    def is_stop_words(self, token):
        r"""
        Judge whether the input word belongs to the stop words vocab.

        :param str token: the input word to be judged
        :return bool: is a stop word or not

        """
        return self.stop_words is not None and token in self.stop_words

    def pre_skip_aug(self, tokens, mask):
        r"""
        Skip the tokens in stop words list or punctuation list.

        :param list tokens: the list of tokens
        :param list mask: the list of mask
                Indicates whether each word is allowed to be substituted.
                ORIGIN is allowed, while TASK_MASK and MODIFIED_MASK is not.
        :return list: List of possible substituted token index.

        """
        assert len(tokens) == len(mask)
        results = []

        for token_idx, token in enumerate(tokens):
            # skip punctuation
            if token in string.punctuation:
                continue
            # skip stopwords by list
            if self.is_stop_words(token):
                continue
            if mask[token_idx] != ORIGIN:
                continue

            results.append(token_idx)

        return results

    def get_trans_cnt(self, size):
        r"""
        Get the num of words/chars transformation.

        :param int size: the size of target sentence
        :return int: number of words to apply transformation.

        """

        cnt = int(self.trans_p * size)

        if cnt < self.trans_min:
            return self.trans_min
        if self.trans_max is not None and cnt > self.trans_max:
            return self.trans_max

        return cnt

    @staticmethod
    def token2chars(word):
        return list(word)

    @staticmethod
    def chars2token(chars):
        assert isinstance(chars, list)

        return ''.join(chars)

r"""
OcrAugTransformation  that apply ocr error simulation to textual input.
============================================

"""

__all__ = ['Ocr']

import random

from ...transformation import WordSubstitute


class OcrRules:
    def __init__(self):
        self.rules = self.get_rules()

    def predict(self, data):
        return self.rules[data]

    # TODO: Read from file
    @classmethod
    def get_rules(cls):
        mapping = {
            '0': ['8', '9', 'o', 'O', 'D'],
            '1': ['4', '7', 'l', 'I'],
            '2': ['z', 'Z'],
            '5': ['8'],
            '6': ['b'],
            '8': ['s', 'S', '@', '&'],
            '9': ['g'],
            'o': ['u'],
            'r': ['k'],
            'C': ['G'],
            'O': ['D', 'U'],
            'E': ['B']
        }

        result = {}

        for k in mapping:
            result[k] = mapping[k]

        for k in mapping:
            for v in mapping[k]:
                if v not in result:
                    result[v] = []

                if k not in result[v]:
                    result[v].append(k)

        return result


class Ocr(WordSubstitute):
    r"""
    Transformation that simulate ocr error by random values.

    """
    def __init__(
        self,
        min_char=1,
        trans_min=1,
        trans_max=10,
        trans_p=0.2,
        stop_words=None,
        **kwargs
    ):
        r"""
        :param int min_char: If word less than this value, do not draw word for
        augmentation
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
            min_char=min_char,
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words)

        self.rules = self.get_rules()

    def __repr__(self):
        return 'Ocr'

    def skip_aug(self, tokens, mask, **kwargs):
        remain_idxes = self.pre_skip_aug(tokens, mask)
        token_idxes = []

        for idx in remain_idxes:
            for char in tokens[idx]:
                if char in self.rules.rules and len(
                        self.rules.predict(char)) > 0:
                    token_idxes.append(idx)
                    break

        return token_idxes

    def _get_candidates(self, word, n=3, **kwargs):
        r"""
        Get a list of transformed tokens.

        :param str word: token word to transform.
        :param int n: number of transformed tokens to generate.
        :param kwargs:
        :return list replaced_tokens: replaced tokens list

        """
        replaced_tokens = []
        chars = self.token2chars(word)
        valid_chars_idxes = [
            idx for idx in range(
                len(chars)) if chars[idx] in self.rules.rules and len(
                self.rules.predict(
                    chars[idx])) > 0]
        if not valid_chars_idxes:
            return []

        # putback sampling
        replace_char_idxes = [
            random.sample(
                valid_chars_idxes,
                1)[0] for i in range(n)]
        replace_idx_dic = {}

        for idx in set(replace_char_idxes):
            replace_idx_dic[idx] = replace_char_idxes.count(idx)

        for replace_idx in replace_idx_dic:
            sample_num = replace_idx_dic[replace_idx]
            cand_chars = self.sample_num(
                self.rules.predict(
                    chars[replace_idx]), sample_num)

            for cand_char in cand_chars:
                replaced_tokens.append(
                    self.chars2token(
                        chars[:replace_idx] + [cand_char] + chars[
                                                            replace_idx + 1:]))

        return replaced_tokens

    @classmethod
    def get_rules(cls):
        return OcrRules()

r"""
KeyboardTransformation  Class
============================================

"""

__all__ = ["Keyboard"]

import os
import re
import random

from ....common.settings import KEYBOARD_RULES
from ...transformation import WordSubstitute
from ....common.utils.install import download_if_needed
from ....common.utils.load import json_loader


class KeyboardRules:
    def __init__(
        self,
        special_char=True,
        numeric=True,
        upper_case=True,
        lang="en",
        rules_path=None
    ):
        self.special_char = special_char
        self.numeric = numeric
        self.upper_case = upper_case
        self.lang = lang
        self.rules = self.get_rules(
            model_path=rules_path,
            special_char=special_char,
            numeric=numeric,
            upper_case=upper_case,
            lang=lang)

    def predict(self, data):
        return self.rules[data]

    # TODO: Extending to 2 keyboard distance
    @classmethod
    def get_rules(
            cls,
            model_path,
            special_char=True,
            numeric=True,
            upper_case=True,
            lang="en"):
        model_path = download_if_needed(model_path)
        if not os.path.exists(model_path):
            raise ValueError(
                'The model_path does not exist. Please check "{}"'.format(
                    model_path))

        mapping = json_loader(model_path)

        result = {}

        for key, values in mapping.items():
            # Skip records if key is numeric while include_numeric is false
            if not numeric and re.match("^[0-9]*$", key):
                continue
            # skip record if key is special character while include_spec is
            # false
            if not special_char and not re.match("^[a-z0-9]*$", key):
                continue

            result[key] = []
            result[key.upper()] = []

            for value in values:
                # Skip record if value is numeric while include_numeric is
                # false
                if not numeric and re.match("^[0-9]*$", value):
                    continue

                # skip record if value is special character while include_spec
                # is false
                if not special_char and not re.match("^[a-z0-9]*$", value):
                    continue

                result[key].append(value)

                if upper_case:
                    result[key].append(value.upper())
                    result[key.upper()].append(value)
                    result[key.upper()].append(value.upper())

        clean_result = {}
        for key, values in result.items():
            # clear empty mapping
            if len(values) == 0:
                continue

            # de-duplicate
            values = [v for v in values if v != key]
            values = sorted(list(set(values)))

            clean_result[key] = values

        return clean_result


class Keyboard(WordSubstitute):
    r"""
    Transformation that simulate typo error by random values.

    https://arxiv.org/pdf/1711.02173.pdf

    For example, people may type i as o incorrectly.\
    One keyboard distance is leveraged to replace character by possible
    keyboard error.

    """
    def __init__(
        self,
        min_char=4,
        trans_min=1,
        trans_max=10,
        trans_p=0.2,
        stop_words=None,
        rules_path=None,
        include_special_char=True,
        include_numeric=True,
        include_upper_case=True,
        lang="en",
        **kwargs
    ):
        r"""
        :param int min_char: If word less than this value, do not draw word for
        augmentation
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
        :param str rules_path: Loading customize model from file system
        :param bool include_special_char: Include special character
        :param bool include_numeric: If True, numeric character may be included
            in augmented data.
        :param bool include_upper_case: If True, upper case character may be
            included in augmented data.
        :param str lang: Indicate built-in language model. Default value is
            'en'. Possible values are 'en' and 'th'. If custom model is used
            (passing model_path), this value will be ignored.

        """
        super().__init__(
            min_char=min_char,
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words
        )

        self.include_special_char = include_special_char
        self.include_numeric = include_numeric
        self.include_upper_case = include_upper_case
        self.include_lower_case = True
        self.lang = lang

        if rules_path is None:
            if lang != 'en':
                raise ValueError(
                    'Only support en now. You may provide the keyboard mapping '
                    'such that we can support "{}"'.format(lang))
            self.rules_path = KEYBOARD_RULES
        else:
            self.rules_path = rules_path

        self.rules = self.get_rules(
            include_special_char,
            include_numeric,
            include_upper_case,
            lang,
            self.rules_path)

    def __repr__(self):
        return 'Keyboard'

    def skip_aug(self, tokens, mask, pos=None):
        return self.pre_skip_aug(tokens, mask)

    def _get_candidates(self, word, n=5, **kwargs):
        r"""
        Get a list of transformed tokens. Default one word replace one char

        :param str word:  token word to transform.
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
    def get_rules(
            cls,
            special_char=True,
            numeric=True,
            upper_case=True,
            lang="en",
            rules_path=None
    ):
        return KeyboardRules(
            special_char=special_char,
            numeric=numeric,
            upper_case=upper_case,
            lang=lang,
            rules_path=rules_path
        )


r"""
Transformation that apply spelling error simulation to textual input.
==========================================================

"""

__all__ = ['SpellingError']

from ...transformation import WordSubstitute
from ....common.settings import SPELLING_ERROR_DIC
from ....common.utils.install import download_if_needed


class SpellingErrorRules:
    def __init__(
        self,
        dict_path,
        include_reverse=True
    ):
        self.dict_path = dict_path
        self.include_reverse = include_reverse

        self._init()

    def _init(self):
        self.rules = {}
        self.read(self.dict_path)

    def read(self, model_path):
        model_path = download_if_needed(model_path)
        with open(model_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                tokens = line.split(' ')
                # Last token include newline separator
                tokens[-1] = tokens[-1].replace('\n', '')

                key = tokens[0]
                values = tokens[1:]

                if key not in self.rules:
                    self.rules[key] = []

                self.rules[key].extend(values)
                # Remove duplicate mapping
                self.rules[key] = list(set(self.rules[key]))
                # Build reverse mapping
                if self.include_reverse:
                    for value in values:
                        if value not in self.rules:
                            self.rules[value] = []
                        if key not in self.rules[value]:
                            self.rules[value].append(key)

    def predict(self, data):
        if data not in self.rules:
            return None

        return self.rules[data]


class SpellingError(WordSubstitute):
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
            stop_words=stop_words
        )

        self.rules_path = rules_path if rules_path else SPELLING_ERROR_DIC
        self.include_reverse = include_reverse
        self.rules = self.get_rules()

    def __repr__(self):
        return 'SpellingError'

    def skip_aug(self, tokens, mask, **kwargs):
        pre_skipped_idxes = self.pre_skip_aug(tokens, mask)
        results = []

        for token_idx in pre_skipped_idxes:
            # Some words do not exit. It will be excluded in lucky draw.
            token = tokens[token_idx]
            if token in self.rules.rules and len(self.rules.rules[token]) > 0:
                results.append(token_idx)

        return results

    def _get_candidates(self, word, n=1, **kwargs):
        r"""
        Get a list of transformed tokens. Default one word replace one char.

        :param str word: token word to transform.
        :param int n: number of transformed tokens to generate.
        :param kwargs:
        :return: candidate list

        """
        return self.sample_num(self.rules.predict(word), n)

    def get_rules(self):
        return SpellingErrorRules(self.rules_path, self.include_reverse)

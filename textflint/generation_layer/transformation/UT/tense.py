r"""
Transform all verb tenses in sentence
==========================================================
"""

__all__ = ['Tense']

from ...transformation import WordSubstitute
from ....common.settings import VERB_PATH, VERB_TAG
from ....common.utils.load import json_loader
from ....common.utils.install import download_if_needed


class Tense(WordSubstitute):
    r"""
    Transforms all verb tenses in sentence.

    Offline Vocabulary is provided.
    Notice: transformed sentence will have syntax errors.

    TODO, align verb tense to avoid grammar error.
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.verb_dic = json_loader(download_if_needed(VERB_PATH))
        self.get_pos = True

    def __repr__(self):
        return 'Tense'

    def skip_aug(self, tokens, mask, pos=None):
        return self.pre_skip_aug(tokens, mask=mask)

    def _get_candidates(self, word, pos=None, n=1):
        r"""
        Returns a list containing all possible words .

        :param str word: token word to transform.
        :param str pos: pos tag
        :param int n: max number of candidates
        :return list candidates: candidates list

        """
        if pos not in VERB_TAG:
            return []
        else:
            candidates = self._get_tense_list(word)
            return self.sample_num(candidates, n)

    def _get_tense_list(self, word):
        if word in self.verb_dic:
            return self.verb_dic[word]
        else:
            return []


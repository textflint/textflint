r"""
Swap/delete/add random character for entities
==========================================================
"""
__all__ = ["EntTypos"]

from ....common.utils.word_op import *
from ..transformation import Transformation
from ....common.utils.list_op import trade_off_sub_words


class EntTypos(Transformation):
    r"""
    Transformation that simulate typos error to transform sentence.

        https://arxiv.org/pdf/1711.02173.pdf

    """

    def __init__(
            self,
            mode="random",
            skip_first_char=True,
            skip_last_char=False,
            **kwargs):
        r"""
        :param str mode: just support
        ['random', 'replace', 'swap', 'insert', 'delete']
        :param bool skip_first_char: whether skip the first char of target word
        :param bool skip_last_char: whether skip the last char of target word.
        :param **kwargs:
        """
        super().__init__()
        self._mode = mode
        self.skip_first_char = skip_first_char
        self.skip_last_char = skip_last_char

    def __repr__(self):
        return 'EntTypos'

    def _transform(self, sample, n=1, **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~NERSample input_sample: Data sample for augmentation
        :param int n: Default is 5. MAx number of unique augmented output
        :return: list of NERSample
        """
        rep_samples = []
        entities = sample.entities[::-1]
        rep_entities = []
        candidates = []

        for entity in entities:
            cur_entity = entity['entity']
            entity_tokens = cur_entity.split(" ")
            rep_idx = random.randint(0, len(entity_tokens) - 1)
            rep_tokens = self._get_replacement_words(
                entity_tokens[rep_idx], n=n)

            if rep_tokens:
                rep_entities.append(entity)
                candidates.append([" ".join(entity_tokens[:rep_idx] +
                                            [rep_token] +
                                            entity_tokens[
                                            rep_idx + 1:]) for
                                   rep_token in rep_tokens])
        candidates, rep_entities = trade_off_sub_words(
            candidates, rep_entities, n=n)

        if not candidates:
            return []

        for i in range(len(candidates)):
            _candidates = candidates[i]
            rep_samples.append(
                sample.entities_replace(
                    rep_entities, _candidates))

        return rep_samples

    def _get_replacement_words(self, word, n=1):
        r"""
        Returns a list of words with typo errors.

        :param string word: the original word to be replaced
        :param int n: number of try times
        :return list: the list of candidate words to replace original words
        """

        candidates = []

        for i in range(n):
            typo_method = self._get_typo_method()
            # default add one typo to each word
            typo_candidate = typo_method(
                word,
                num=1,
                skip_first=self.skip_first_char,
                skip_last=self.skip_last_char)
            if typo_candidate == word:
                typo_method = random.choice([replace, insert, delete])
                typo_candidate = typo_method(
                    word,
                    num=1,
                    skip_first=self.skip_first_char,
                    skip_last=self.skip_last_char)
            if typo_candidate:
                candidates.append(typo_candidate)

        return list(candidates)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode_value):
        assert mode_value in ['random', 'replace', 'swap', 'insert', 'delete']
        self._mode = mode_value

    def _get_typo_method(self):
        if self._mode == 'replace':
            return replace
        elif self._mode == 'swap':
            return swap
        elif self._mode == 'insert':
            return insert
        elif self._mode == 'delete':
            return delete
        else:
            return random.choice([replace, swap, insert, delete])

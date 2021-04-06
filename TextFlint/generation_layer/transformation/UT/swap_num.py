r"""
Swapping numbers in sentences
==========================================================
"""

__all__ = ['SwapNum']

import random

from ...transformation import WordSubstitute


class SwapNum(WordSubstitute):
    r"""
    Transforms an input by replacing its numbers.

    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

    def __repr__(self):
        return 'SwapNum'

    def _get_candidates(self, word, pos=None, n=1):
        r"""
        Returns a list containing all possible words .

        :param str word: token word to transform.
        :param str pos: pos tag
        :param int n: max number of candidates
        :return list candidates: candidates list

        """
        n = max(n, 1)
        candidates = []

        if word.isdigit():
            number = int(word)

            while True:
                candidate = str(random.randint(0, (number + 1) * n))

                if int(candidate) != number:
                    candidates.append(candidate)

                if len(candidates) >= n:
                    break

        return candidates

    def skip_aug(self, tokens, mask, **kwargs):
        return self.pre_skip_aug(tokens, mask)


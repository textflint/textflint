r"""
Swapping Arabic Numbers in parallel sentences
==========================================================
"""

__all__ = ['SwapParallelNum']

import random

from .parallel_word_substitute import ParallelWordSubstitute
 

class SwapParallelNum(ParallelWordSubstitute):
    r"""
    Transforms parallelly an input by replacing its Arabic Numbers.

    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

    def __repr__(self):
        return 'SwapParallelNum'

    def _get_candidates(self, word, target, pos=None, n=1):
        r"""
        Returns a list containing all possible words .

        :param str word: token word to transform.
        :param str pos: pos tag
        :param int n: max number of candidates
        :return list candidates: candidates list

        """
        n = max(n, 1)
        last_target_index = target[2]
        candidates = []
        target_candidate_index = None
        if word.isdigit():
            number = int(word)
            for index in range(last_target_index, len(target[0])):
                target_word = target[0][index]
                if word == target_word:
                    while True:
                        candidate = str(random.randint(0, (number + 1) * n))

                        if int(candidate) != number and (int(candidate) not in candidates):
                            candidates.append(candidate)

                        if len(candidates) >= n:
                            break
                    target_candidate_index = target[1][index]
                    break
        return candidates, target_candidate_index

    def skip_aug(self, tokens, mask, **kwargs):
        return self.pre_skip_aug(tokens, mask)
r"""
Swapping same word in parallel sentences
==========================================================
"""

__all__ = ['SwapParallelSameWord']

import random
import string

from .parallel_word_substitute import ParallelWordSubstitute
 

class SwapParallelSameWord(ParallelWordSubstitute):
    r"""
    Transforms parallelly an input by replacing the same word.

    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

    def __repr__(self):
        return 'SwapParallelSameWord'

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
        if not word.isdigit():
            for index in range(last_target_index, len(target[0])):
                target_word = target[0][index]
                if word == target_word:
                    while True:
                        candidate = self.get_random_word(word)

                        if candidate not in candidates:
                            candidates.append(candidate)

                        if len(candidates) >= n:
                            break
                    target_candidate_index = target[1][index]
                    break

        return candidates, target_candidate_index

    def skip_aug(self, tokens, mask, **kwargs):
        return self.pre_skip_aug(tokens, mask)

    def get_random_word(self, word):
        r"""
        Random return a word.

        :param int length: length of the word.
        :return str candidate: random word.

        """
        length = len(word)
        candidate = ''
        min_len = 3
        max_len = length + 3
        candidate_len = random.randint(min_len, max_len)
        if word.istitle():
            candidate += random.choice(string.ascii_uppercase)
            for i in range(1, candidate_len):
                candidate += random.choice(string.ascii_lowercase)
        elif word.isupper():
            for i in range(candidate_len):
                candidate += random.choice(string.ascii_uppercase)
        else:
            for i in range(candidate_len):
                candidate += random.choice(string.ascii_lowercase)
        return candidate
        
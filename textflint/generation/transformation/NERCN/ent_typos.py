r"""
Swap/replace random character for entities
==========================================================
"""
__all__ = ["EntTypos"]

import string
from string import ascii_lowercase
from ....common.utils.word_op import *
from ..transformation import Transformation
from ....common.utils.list_op import trade_off_sub_words
from textflint.generation.transformation.UTCN import CnHomophones

def _get_random_character():
    """
    :return: return a random charactor
    """
    return random.choice(string.ascii_letters+string.digits)
def _get_homophone(char):
    ch = CnHomophones(get_pos=True)
    return ch.homophones(char, n=1)[0]

# def _insert_cn(word, num=1, skip_first=True, skip_last=False):
#     """
#     Perturb the word with 1 random chinese character inserted.

#     :param str word: word to be inserted
#     :param int num: number of typos to add
#     :param bool skip_first: whether insert char at the beginning of word
#     :param bool skip_last: whether insert char at the end of word
#     :return: perturbed strings
#     """
#     if len(word) <= 1:
#         return word

#     chars = list(word)
#     start, end = get_start_end(word, skip_first, skip_last)

#     if end - start + 2 < num:
#         return None

#     swap_idxes = random.sample(list(range(start, end + 2)), num)
#     swap_idxes.sort(reverse=True)

#     for idx in swap_idxes:
#         insert_char = _get_random_character()
#         chars = chars[:idx] + [insert_char] + chars[idx:]

#     return "".join(chars)

def _replace_cn(word, num=1, skip_first=True, skip_last=False):
    """
    Perturb the word with 1 letter substituted for a random letter.

    :param str word: target word
    :param int num: number of typos to add
    :param bool skip_first: whether replace the char at the beginning of word
    :param bool skip_last: whether replace the char at the beginning of word
    :return: perturbed strings
    """
    if len(word) <= 1:
        return []

    chars = list(word)
    start, end = get_start_end(word, skip_first, skip_last)

    # error swap num, return original word
    if end - start + 1 < num:
        return word

    idxes = random.sample(list(range(start, end + 1)), num)

    for idx in idxes:
        chars[idx] = _get_homophone(chars[idx])

    return "".join(chars)


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
        **kwargs
    ):
        r"""
        :param str mode: just support
        ['random', 'replace', 'swap']
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
        entities = sample.entities
        rep_entities = []
        candidates = []

        for entity in entities:
            cur_entity = entity['entity']

            rep_token = self._get_replacement_words(
                cur_entity, n=n
            )

            if rep_token:
                rep_entities.append(entity)
                candidates.append(rep_token)

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
                typo_method = random.choice([_replace_cn])
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
        assert mode_value in ['random', 'replace', 'swap']
        self._mode = mode_value

    def _get_typo_method(self):
        if self._mode == 'replace':
            return _replace_cn
        elif self._mode == 'swap':
            return swap

        else:
            return random.choice([_replace_cn, swap])



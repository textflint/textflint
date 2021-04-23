r"""
contract or extend sentence by common abbreviations
==========================================================
"""

__all__ = ['Contraction']

from ...transformation import Transformation
from ....common.settings import CONTRACTION_PHRASES


class Contraction(Transformation):
    r""" Transforms input by common abbreviations.

    Each sample generate one transformed sample at most.

    Example::

           "we're playing ping pang ball, you are so lazy. She's so beautiful!"

        >> "we are playing ping pang ball, you're so lazy. She is so beautiful!"

    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.phrases = CONTRACTION_PHRASES
        self.contractions = {v: k for k, v in CONTRACTION_PHRASES.items()}

    def __repr__(self):
        return 'Contraction'

    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform text string according transform_field.

        :param ~Sample sample: input data, normally one data component.
        :param str field: indicate which field to transform.
        :param int n: number of generated samples
        :param kwargs:
        :return list trans_samples: transformed sample list.

        """
        tokens = sample.get_words(field)

        contractions_indices, phrases = self._get_contractions(tokens)
        phrases_indices, contractions = self._get_expanded_phrases(tokens)
        indices = contractions_indices + phrases_indices
        rep_items = phrases + contractions

        if indices:
            return [
                sample.unequal_replace_field_at_indices(
                    field, indices, rep_items)]

        return []

    def _get_contractions(self, tokens):
        r"""
        Get contractions string in given tokens. this function work while
        tokenizer won't split contract string like 'can't' -> 'can ' t' or
        'can 't'

        :param list tokens: word list
        :return list indices_list: indices of contract strings
        :return list phrases_list: expanded phrases.
        """
        indices_list = []
        phrases_list = []

        # just judge [:-2] range to avoid Exception
        for index, token in enumerate(tokens[:-2]):
            sub_string = token + tokens[index + 1]
            if sub_string in self.contractions:
                indices_list.append([index, index + 2])
                phrases_list.append(self.contractions[sub_string])

        return indices_list, phrases_list

    def _get_expanded_phrases(self, tokens):
        r"""
        Get expanded phrases which can convert to contractions.

        :param list tokens: word list
        :return list indices_list:  indices of expanded phrase
        :return list contractions_list: expanded phrases, contractions
        """
        indices_list = []
        contractions_list = []
        # avoid collision
        skip = False

        for i in range(len(tokens)):
            # one sublist less judge to avoid Exception
            if len(tokens) > i + 2:
                if skip:
                    skip = False
                    continue
                phrase_string = " ".join(tokens[i: i + 2])

                if phrase_string in self.phrases:
                    skip = True
                    indices_list.append([i, i + 2])
                    contractions_list.append(self.phrases[phrase_string])

        return indices_list, contractions_list

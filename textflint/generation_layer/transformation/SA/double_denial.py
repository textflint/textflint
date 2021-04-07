r"""
Word Swap by swapping word to double denial forms
==========================================================
"""
__all__ = ["DoubleDenial"]
from ..transformation import Transformation
from ....common.settings import SA_DOUBLE_DENIAL_DICT


class DoubleDenial(Transformation):
    r"""
    Transforms an input by replacing its words with double denial forms.

    Example::

        ori: The leading actor is good
        trans: The leading actor is not bad
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.polarity_dict = SA_DOUBLE_DENIAL_DICT

    def __repr__(self):
        return 'DoubleDenial'

    def _transform(self, sample, n=1, **kwargs):
        r"""
        Transform text string according field, this kind of transformation
        can only produce one sample.

        :param ~SASample sample: input data, a SASample contains 'x' field
            and 'y' field
        :param int n: number of generated samples, this transformation can
            only generate one sample
        :return list trans_samples: transformed sample list that only contain
            one sample

        """
        tokens = sample.get_words('x')
        sub_indices, sub_words = self._get_double_denial_info(tokens)

        if not sub_indices:
            return []

        sub_tokens = [self.processor.tokenize(sub_word)
                      for sub_word in sub_words]
        sample = sample.unequal_replace_field_at_indices(
            'x', sub_indices, sub_tokens)
        trans_samples = [sample]
        return trans_samples

    def _get_double_denial_info(self, tokens):
        r"""
        get words that can be converted to a double denial form

        :param list tokens: tokenized words
        :return list indices:  indices of tokens that should be replaced
        :return list double_denial_words: The new words that correspond to
            indices and is used to replace them

        """
        if tokens is str:
            tokens = [tokens]

        tokens = [token.lower() for token in tokens if isinstance(token, str)]
        indices = []
        double_denial_words = []

        for polarity_word in self.polarity_dict.keys():
            polarity_word_cnt = tokens.count(polarity_word)
            current_index = -1
            for i in range(polarity_word_cnt):
                # TODO
                try:
                    current_index = tokens.index(polarity_word, current_index+1)
                    indices.append(current_index)
                    double_denial_words.append(
                        self.polarity_dict[polarity_word])
                except ValueError:
                    pass

        return indices, double_denial_words

r"""
Replacing its words with antonyms provided by WordNet
==========================================================
"""

from nltk.wsd import lesk
from ..transformation import Transformation
from ....common.settings import BLACK_LIST_WORD

__all__ = ['SwapAnt']


class SwapAnt(Transformation):
    r"""
    Transforms an input by replacing its words with antonyms provided by
    WordNet. Download nltk_data before running.

    Implement follow by Stress Test Evaluation for Natural Language Inference
    For the correctness of trasformation we swap the word has best_sense(
    Wordnet) to its antonym

    https://www.aclweb.org/anthology/C18-1198/

    exmaple:
    {
        hypothesis: I hate this book.
        premise: This book is my favorite.
        label: contradiction
    }
    """

    def __init__(
        self,
        language="eng"
    ):
        r"""
        :param string language: language of transformation
        """
        super().__init__()
        self.language = language
        self.blacklist_words = BLACK_LIST_WORD

    def __repr__(self):
        return 'SwapAnt'

    def transform(self, sample, n=1, **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~NLISample sample: Data sample for augmentation
        :param int n: Default is 1. MAX number of unique augmented output
        :param **kwargs:
        :return: Augmented data
        """
        transform_results = self._transform(sample, **kwargs)

        if transform_results:
            return [data for data in transform_results if not data.is_origin]
        else:
            return []

    def _transform(self, sample, **kwargs):
        r"""
        Transform text string, this kind of transformation
        can only produce one sample.

        :param ~NLISample sample: input data, a NLISample contains 'hypothesis'
            field, 'premise' field and 'y' field
        :param int n: number of generated samples, this transformation can only
            generate one sample
        :return list trans_samples: transformed sample list that only contain
            one sample
        """

        label_tag = sample.get_value('y')

        if label_tag != 'entailment':
            return None

        tokens1 = sample.get_words('hypothesis')
        tokens2 = sample.get_words('premise')
        original_text2 = sample.get_value('premise')

        for num, each_word in enumerate(tokens2):
            if each_word not in self.blacklist_words:
                # todo， pre_process 包实现
                best_sense = lesk(tokens2, each_word)
                if best_sense is not None and (
                        best_sense.pos() == 's' or best_sense.pos() == 'n'):
                    for lemma in best_sense.lemmas():
                        possible_antonyms = lemma.antonyms()

                        for antonym in possible_antonyms:
                            if "_" in antonym._name or antonym._name == \
                                    "civilian":
                                continue
                            if each_word not in tokens1:
                                continue

                            new_s1 = original_text2.replace(
                                each_word, antonym._name, 1)

                            sample = sample.replace_fields(
                                ['hypothesis', 'y'], [new_s1, 'contradiction']
                            )
        return [sample]

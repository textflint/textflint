r"""
Reverse the polarity of target aspect in ABSA task
==========================================================
"""

from copy import deepcopy
from .absa_transformation import ABSATransformation

__all__ = ['RevTgt']


class RevTgt(ABSATransformation):
    r"""
    Transforms the polarity of target by replacing its opinion words
    with antonyms provided by WordNet or adding the negation that
    pre-defined in our negative word list.

    Example::

        Original sentence: "BEST spicy tuna roll, great asian salad.
        (Target: spicy tuna roll)"
        Transformed sentence: "BAD spicy tuna roll, great asian salad."
    """

    def __init__(
        self,
        language="eng"
    ):
        super().__init__()
        if language != "eng":
            raise ValueError(f"Language {language} is not available.")
        self.language = language
        self.tokenize = self.processor.tokenize

    def __repr__(self):
        return "RevTgt"

    def _transform(self, sample, n=1, field='sentence', **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~textflint.ABSAsample sample: input ABSAsample
        :param int n: the number of transformation,
            in ABSA-specific transformations n=1
        :param str field:field name
        :return list: list of transformed ABSAsample

        """
        trans_samples = []
        self.terms = sample.terms
        self.sentence = sample.sentence.text
        self.words_list = sample.sentence.words
        self.term_list = sample.term_list
        if sample.trans_id is None:
            self.trans_id = [idx for idx in self.terms]
        else:
            self.trans_id = [sample.trans_id]

        for term_id in self.trans_id:
            trans_sample = sample.clone(sample)
            trans_words, trans_polarity, trans_terms = \
                self._trans_aspect_polarity(
                    term_id)
            trans_words = self._trans_conjunction(
                term_id, trans_words, trans_polarity)
            trans_sentence = self.get_sentence(trans_words, self.sentence)
            trans_sample.update_sentence(trans_sentence)
            trans_sample.update_terms(trans_terms)
            trans_sample.update_term_list(trans_sample)
            trans_sample.trans_id = term_id
            trans_samples.append(trans_sample)

        return trans_samples

    def _trans_aspect_polarity(self, term_id):
        r"""
        Reverse the polarity of the aspect.

        :param str term_id: term id
        :return: opinion words, polarity, terms of transformed aspect

        """
        aspect_term = self.terms[term_id]
        words_list = self.words_list
        trans_terms = deepcopy(self.terms)
        opinion_position = self.terms[term_id]['opinion_position']
        aspect_polarity = aspect_term['polarity']

        if aspect_polarity == 'positive':
            trans_words, trans_opinion_words = self.reverse(
                words_list, opinion_position)
            trans_terms[term_id]['polarity'] = 'negative'
        elif aspect_polarity == 'negative':
            trans_words, trans_opinion_words = self.reverse(
                words_list, opinion_position)
            trans_terms[term_id]['polarity'] = 'positive'
        else:
            trans_words, trans_opinion_words = self.reverse(
                words_list, opinion_position)
            trans_terms[term_id]['polarity'] = 'neutral'
        trans_polarity = trans_terms[term_id]['polarity']
        trans_words, trans_terms = self.update_sentence_terms(
            trans_words, trans_terms, trans_opinion_words, opinion_position)

        return trans_words, trans_polarity, trans_terms

    def _trans_conjunction(self, term_id, trans_words, trans_polarity):
        r"""
        Transform the conjunction words in sentence.

        :param str term_id: term id
        :param list trans_words: tokenized words of transformed sentence
        :param str trans_polarity: transformed polarity
        :return list: tokenized words of transformed sentence

        """
        term_list = self.term_list
        aspect_opinions = set()
        conjunction_list = ['and', 'but']
        aspect_opinions.add(aspect_opi for aspect_opi in
                            term_list[term_id]['opinion_words'])
        other_opinions, other_polarity = \
            self.get_other_opinions(term_list, term_id)

        if len(other_polarity) > 0 and len(
                aspect_opinions & other_opinions) == 0:
            conjunction_idx = self.get_conjunction_idx(
                trans_words, term_list[term_id], conjunction_list)
            if conjunction_idx is not None and trans_polarity \
                    not in other_polarity:
                trans_words[conjunction_idx] = 'but'
            elif conjunction_idx is not None and \
                    trans_polarity in other_polarity:
                trans_words[conjunction_idx] = 'and'

        return trans_words

    @staticmethod
    def get_other_opinions(term_to_position_list, term_id):
        r"""
        Get the polarity of other opinions.

        :param dict term_to_position_list: position list of terms
        :param str term_id: term id
        :return set: return opinions and polarities of other terms

        """
        other_polarity = set()
        other_opinions = set()

        for other_term_id in term_to_position_list:
            if other_term_id != term_id:
                other_polarity.add(
                    term_to_position_list[other_term_id]['polarity'])
                for other_opi in term_to_position_list[other_term_id][
                    'opinion_words']:
                    other_opinions.add(other_opi[0])

        return other_opinions, other_polarity

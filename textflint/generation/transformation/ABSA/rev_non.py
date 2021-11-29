r"""
Reverse the polarity of non-target in ABSA task
==========================================================
"""
from copy import deepcopy
from ...transformation.ABSA.absa_transformation import ABSATransformation
__all__ = ['RevNon']


class RevNon(ABSATransformation):
    r"""
    Transforms the polarity of non-target by replacing its opinion words
        with antonyms provided by WordNet or adding the negation that
        pre-defined in our negative word list.

    Example::

        Original sentence: "BEST spicy tuna roll, great asian salad.
        （Target: spicy tuna roll)"
        Transformed sentence: "BEST spicy tuna roll, not great asian salad."

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
        return "RevNon"

    def _transform(self, sample, n=1, field='sentence', **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~textflint.input.component.sample.ABSAsample sample: input ABSAsample
        :param int n: the number of transformation, in
            ABSA-specific transformations n=1
        :param str field:field name
        :return list: list of transformed ABSAsample

        """
        trans_samples = []
        self.sentence = sample.sentence.text
        self.words_list = sample.sentence.words
        self.terms = sample.terms
        self.term_list = sample.term_list
        if sample.trans_id is None:
            self.trans_id = [idx for idx in self.terms]
        else:
            self.trans_id = [sample.trans_id]

        for term_id in self.trans_id:
            other_id_list = [idx for idx in self.terms]
            other_id_list.remove(term_id)
            if len(other_id_list) != 0:
                trans_sample = sample.clone(sample)
                trans_words, trans_terms = self._trans_other_polarity(term_id)
                trans_sentence = self.get_sentence(trans_words, self.sentence)
                trans_sample.update_sentence(trans_sentence)
                trans_sample.update_terms(trans_terms)
                trans_sample.update_term_list(trans_sample)
                trans_sample.trans_id = term_id
                trans_samples.append(trans_sample)
        if trans_samples:

            return trans_samples

    def _trans_other_polarity(self, term_id):
        r"""
        Transform the polarity of other opinions.

        :param str term_id: term id
        :return: tokenized words and terms of transformed sentence

        """
        terms = self.terms
        aspect_term = terms[term_id]
        aspect_polarity = aspect_term['polarity']
        other_id_list = [idx for idx in terms]
        other_id_list.remove(term_id)
        reverse_list = []
        exaggerate_list = []
        non_overlap_opinion = []

        for other_index, other_id in enumerate(other_id_list):
            other_term = terms[other_id]
            other_opinion = terms[other_id]['opinion_position']
            term_polarity = other_term['polarity']

            for opinion in other_opinion:
                if opinion not in non_overlap_opinion:
                    non_overlap_opinion.append(opinion)
                    if aspect_polarity == term_polarity in [
                            'positive', 'negative']:
                        reverse_list.append(opinion)
                    else:
                        exaggerate_list.append(opinion)
            if len(non_overlap_opinion) == 0:
                continue

        trans_words, trans_terms = self._trans_term_polarity(
            term_id, reverse_list, exaggerate_list)

        return trans_words, trans_terms

    def _trans_term_polarity(self, term_id, reverse_list, exaggerate_list):
        r"""
        Transform the polarity of a certain term.

        :param str term_id: term id
        :param list reverse_list: pre-defined reverse_list
        :param list exaggerate_list: pre-defined exaggerate_list
        :return: tokenized words and terms of transformed sentence

        """
        trans_terms = deepcopy(self.terms)
        other_id_list = [idx for idx in trans_terms]
        other_id_list.remove(term_id)
        trans_words = self.words_list
        aspect_term = trans_terms[term_id]
        trans_opinion_reverse = []
        trans_opinion_exaggerate = []

        if len(reverse_list) != 0:
            trans_words, trans_opinion_reverse = self.reverse(
                trans_words, reverse_list)
            trans_words = self._trans_conjunction(aspect_term, trans_words)
        if len(exaggerate_list) != 0:
            trans_words, trans_opinion_exaggerate = self.exaggerate(
                trans_words, exaggerate_list)
        trans_position = reverse_list + exaggerate_list
        trans_opinion = trans_opinion_reverse + trans_opinion_exaggerate

        for idx in other_id_list:
            if trans_terms[idx]['polarity'] == 'positive':
                trans_terms[idx]['polarity'] = 'negative'
            elif trans_terms[idx]['polarity'] == 'negative':
                trans_terms[idx]['polarity'] = 'positive'
            else:
                trans_terms[idx]['polarity'] = 'neutral'

        trans_words, trans_terms = self.update_sentence_terms(
            trans_words, trans_terms, trans_opinion, trans_position)

        return trans_words, trans_terms

    def _trans_conjunction(self, aspect_term, trans_words):
        r"""
        Transform the conjunction words in sentence.

        :param dict aspect_term: aspect term
        :param list trans_words: tokenized words and terms of
            transformed sentence
        :return list: tokenized words and terms of transformed sentence

        """
        conjunction_list = ['and']
        conjunction_idx = self.get_conjunction_idx(
            trans_words, aspect_term, conjunction_list)
        if conjunction_idx is not None:
            trans_words[conjunction_idx] = 'but'
        return trans_words

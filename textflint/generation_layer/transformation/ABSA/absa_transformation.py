import random
from abc import ABC
from copy import deepcopy

from ..transformation import Transformation
from ....common.settings import NEGATIVE_WORDS_LIST, DEGREE_WORD_LIST
__all__ = ['ABSATransformation']


class ABSATransformation(Transformation, ABC):
    r"""
    An class that supply methods for ABSA task data transformation.

    """

    def __init__(self):
        super().__init__()

        self.negative_words_list = sorted(NEGATIVE_WORDS_LIST,
                                          key=lambda s: len(s), reverse=True)
        self.tokenize = self.processor.tokenize
        self.untokenize = self.processor.inverse_tokenize
        self.get_antonyms = self.processor.get_antonyms

    def reverse(self, words_list, opinion_position):
        r"""
        Reverse the polarity of opinions.

        :param list words_list: tokenized words of original sentence
        :param list opinion_position: opinion position
        :return: transformed sentence and transformed opinion words
        """
        trans_words = deepcopy(words_list)
        trans_opinion_words = []

        for position in opinion_position:
            opinion_from = position[0]
            opinion_to = position[1]
            opinion_list = trans_words[opinion_from:opinion_to]
            trans_words, opinion_words, opinion_from, opinion_to, has_neg = \
                self.check_negation(trans_words, opinion_from, opinion_to)
            if len(opinion_list) == 1:
                trans_words, trans_opinion_words = self.reverse_opinion(
                    trans_words, trans_opinion_words, opinion_from, opinion_to,
                    has_neg)
            elif len(opinion_list) > 1:
                if has_neg:
                    trans_opinion_words.append(
                        [opinion_from, opinion_to,
                         self.untokenize(opinion_words)])
                else:
                    # negate the closest verb
                    trans_opinion_words.append(
                        [opinion_from, opinion_to, self.untokenize(
                            ['not ' + opinion_words[0]] + opinion_words[1:])])
                    trans_words[opinion_from:opinion_from +
                                1] = ['not ' + opinion_words[0]]

        return trans_words, trans_opinion_words

    def exaggerate(self, words_list, opinions):
        r"""
        Exaggerate the opinion words.

        :param list words_list: tokenized words of original sentence
        :param list opinions: opinion words and their positions
        :return: transformed sentence and opinion words
        """
        new_words = deepcopy(words_list)
        new_opi_words = []

        for i in range(len(opinions)):
            opi_position = opinions[i]
            opi_from = opi_position[0]
            opi_to = opi_position[1]
            new_words, new_opi = self.add_degree_words(
                new_words, opi_from, opi_to)
            new_opi_words.append([opi_from, opi_to, self.untokenize(new_opi)])

        return new_words, new_opi_words

    def get_postag(self, sentence, start, end):
        r"""
        Get the postag.

        :param list|str sentence: sentence
        :param int start: start index
        :param int end: end index
        :return list: postag
        """

        tags = self.processor.get_pos(sentence)
        if end != -1:
            return tags[start:end]
        else:
            return tags[start:]

    def refine_candidate(self, trans_words, opi_from, opi_to, candidate_list):
        r"""
        Refine the candidate opinion words.

        :param list trans_words: tokenized words of transformed sentence
        :param int opi_from: start position of opinion words
        :param int opi_to: end position of opinion words
        :param set candidate_list: candidate antonyms word list
        :return list: refined candidate word list
        """
        if len(trans_words) == 0:
            return []
        postag_list = self.get_postag(trans_words, 0, -1)
        postag_list = [t[1] for t in postag_list]
        refined_candi = self.get_candidate(
            candidate_list, trans_words, postag_list, opi_from, opi_to)

        return refined_candi

    @staticmethod
    def get_word2id(text, lower=True):
        r"""
        Get the index of words in sentence.

        :param str text: input text
        :param bool lower: whether text is lowercase or not
        :return dict: index of words
        """
        word2idx = {}
        idx = 1
        if lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1

        return word2idx

    @staticmethod
    def add_degree_words(word_list, from_idx, to_idx):
        r"""
        Add the degree words to sentence.

        :param list word_list: tokenized words of original sentence
        :param int from_idx: index of start
        :param int to_idx: index of end
        :return: transformed sentence and opinion words
        """
        candidate_list = DEGREE_WORD_LIST
        select = random.randint(0, len(candidate_list) - 1)
        opi1 = [' '.join([candidate_list[select]] +
                         word_list[from_idx: from_idx + 1])]
        new_words = word_list[:from_idx] + opi1 + word_list[from_idx + 1:]
        opi = new_words[from_idx: to_idx]

        return new_words, opi

    @staticmethod
    def get_conjunction_idx(trans_words, aspect_term, conjunction_list):
        r"""
        Get the index of conjunction words in  conjunction_list.

        :param list trans_words: tokenized words of transformed sentence
        :param dict aspect_term: aspect term
        :param list conjunction_list: conjunction list
        :return list: index of transformed conjunction word
        """
        conjunction_idx = []
        trans_idx = None
        term = aspect_term['term']
        term_from = aspect_term['from']
        term_to = aspect_term['to']
        distance_to_term = len(trans_words)

        for idx, word in enumerate(trans_words):
            if word.lower() in conjunction_list and word.lower() \
                    not in term.lower():
                conjunction_idx.append(idx)
        for idx in conjunction_idx:
            if idx > term_to and idx - term_to < distance_to_term:
                distance_to_term = idx - term_to
                trans_idx = idx
            if idx < term_from and term_to - idx:
                distance_to_term = term_to - idx
                trans_idx = idx

        return trans_idx

    def get_sentence(self, trans_words, sentence):
        r"""
        Untokenize and uppercase to get an output sentence.

        :param list trans_words: transformed sentence
        :param list sentence: original sentence
        :return list: transformed sentence
        """
        trans_sentence = self.untokenize(trans_words)
        if sentence[0].isupper():
            trans_sentence = trans_sentence[0].upper() + trans_sentence[1:]
        return trans_sentence

    def get_term_span(self, trans_sentence, term):
        r"""
        Get the span of term in trans_sentence.

        :param list trans_sentence: transformed sentence
        :param list term: target term
        :return: start and end index of target term
        """
        span_from = 0
        char_from = 0
        char_sentence = ''.join(self.tokenize(trans_sentence))
        char_term = ''.join(self.tokenize(term))
        for idx in range(len(char_sentence)):
            if char_sentence[idx:idx + len(char_term)] == char_term:
                char_from = len(char_sentence[:idx])
                break
        trans_from = 0
        for idx in range(len(trans_sentence)):
            if trans_sentence[idx] != ' ':
                trans_from += 1
            if trans_from == char_from and char_from != 0 and \
                    trans_sentence[idx + 1] != ' ':
                span_from = idx + 1
                break
            if trans_from == char_from and char_from != 0 and \
                    trans_sentence[idx + 1] == ' ':
                span_from = idx + 2
                break
        span_to = span_from + len(term)

        return span_from, span_to

    def get_candidate(
            self,
            candidate_list,
            words_list,
            postag_list,
            opi_from,
            opi_to):
        r"""
        Get the candidate opinion words from words_list.

        :param set candidate_list: candidate words
        :param list words_list: tokenized words of original sentence
        :param list postag_list: postag
        :param int opi_from: start index of opinion
        :param int opi_to: end index of opinion
        :return list: refined candidate words
        """
        refined_candi = []

        for candidate in candidate_list:
            opi = words_list[opi_from:opi_to][0]
            isupper = opi[0].isupper()
            allupper = opi.isupper()
            if allupper:
                candidate = candidate.upper()
            elif isupper:
                candidate = candidate[0].upper() + candidate[1:]
            if opi_from == 0:
                candidate = candidate[0].upper() + candidate[1:]

            new_words = words_list[:opi_from] + \
                [candidate] + words_list[opi_to:]
            # check pos tag
            new_postag_list = self.get_postag(new_words, 0, -1)
            new_postag_list = [t[1] for t in new_postag_list]

            if len([i for i, j in zip(postag_list[opi_from:opi_to],
                                      new_postag_list[opi_from:opi_to]) if
                    i != j]) != 0:
                continue
            refined_candi.append(candidate)

        return refined_candi

    def check_negation(self, trans_words, opinion_from, opinion_to):
        r"""
        Check the negation words in trans_words and delete them.

        :param list trans_words: tokenized words of transformed sentence
        :param int opinion_from: start index of opinion
        :param int opinion_to: end index of opinion
        :return: transformed words, opinion words, position of opinion, and
        whether exist negation in transformed sentence
        """
        opinion_words = trans_words[opinion_from: opinion_to]
        has_neg = False

        for w in self.negative_words_list:
            ws = self.tokenize(w)
            for j in range(opinion_from, opinion_to - len(ws) + 1):
                trans_words_ = ' '.join(trans_words[j:j + len(ws)])
                ws_ = ' '.join(ws)
                if trans_words_.lower() == ws_.lower():
                    trans_words[j: j + len(ws)] = ['DELETE'] * len(ws)
                    has_neg = True
                    opinion_words = trans_words[opinion_from: opinion_to]
                    break

            if has_neg:
                opinion_words.remove('DELETE')
                break

        return trans_words, opinion_words, opinion_from, opinion_to, has_neg

    def reverse_opinion(
            self,
            trans_words,
            trans_opinion_words,
            opinion_from,
            opinion_to,
            has_neg):
        r"""
        Reverse the polarity of original opinion and return the new
        transformed opinion words.

        :param list trans_words: tokenized words of transformed sentence
        :param list trans_opinion_words: transformed opinion words
        :param int opinion_from: start index of opinion
        :param int opinion_to: end index of opinion
        :param bool has_neg: whether exist negation in transformed sentence
        """
        opinion_list = trans_words[opinion_from:opinion_to]
        opinion_words = trans_words[opinion_from:opinion_to]
        opi = opinion_list[0]
        trans_opinion_word = None
        from_to = []

        if has_neg and [opinion_from, opinion_to] not in from_to:
            trans_opinion_word = [
                opinion_from,
                opinion_to,
                self.untokenize(opinion_words)]
        elif [opinion_from, opinion_to] not in from_to:
            opi_pos = self.get_postag(trans_words, opinion_from, opinion_to)
            antonyms = self.get_antonyms(opi_pos)[0]
            candidate = set()
            for antonym in antonyms:
                for ant_word in antonym.lemma_names(lang='eng'):
                    if (
                            (ant_word != opi)
                            and ("_" not in ant_word)
                    ):
                        candidate.add(ant_word)

            refined_candidate = self.refine_candidate(
                trans_words, opinion_from, opinion_to, candidate)
            if len(refined_candidate) == 0:
                trans_opinion_word = [opinion_from, opinion_to,
                                      self.untokenize(['not', opi])]
            else:
                select = random.randint(0, len(refined_candidate) - 1)
                trans_opinion_word = [opinion_from,
                                      opinion_to,
                                      self.untokenize(
                                          [refined_candidate[select]])]
        if trans_opinion_word is not None:
            trans_opinion_words.append(trans_opinion_word)
            from_to.append([opinion_from, opinion_to])
            trans_words[opinion_from: opinion_to] = [trans_opinion_word[2]]

        return trans_words, trans_opinion_words

    def update_sentence_terms(
            self,
            trans_words,
            trans_terms,
            trans_opinion_words,
            opinion_position):
        r"""
        Update the terms and sentence.

        :param list trans_words: tokenized words of transformed sentence
        :param dict trans_terms: transformed terms
        :param list trans_opinion_words: transformed opinion words
        :param list opinion_position: opinion position
        :return: transformed sentence and transformed terms
        """
        terms = deepcopy(trans_terms)
        trans_opinion_position = deepcopy(opinion_position)

        for trans_id, trans_opi in enumerate(trans_opinion_words):
            offset = len(self.tokenize(
                trans_opi[2])) - (trans_opi[1] - trans_opi[0])
            for opi_id, opi in enumerate(opinion_position):
                if opi[0] > trans_opi[0]:
                    trans_opinion_position[opi_id][0] += offset
                    trans_opinion_position[opi_id][1] += offset
            for term_id in terms:
                if terms[term_id]['from'] >= trans_opi[0]:
                    trans_terms[term_id]['from'] += offset
                    trans_terms[term_id]['to'] += offset
                positions = terms[term_id]['opinion_position']
                for pos_id, position in enumerate(positions):
                    if position == opinion_position[trans_id]:
                        trans_terms[term_id]['opinion_words'][pos_id] = \
                            trans_opi[2]
                        trans_terms[term_id]['opinion_position'][pos_id] = [
                            trans_opinion_position[trans_id][0],
                            trans_opinion_position[trans_id][1] + offset]
                    elif position[0] > trans_opi[0]:
                        trans_terms[term_id]['opinion_position'][pos_id][0] += \
                            offset
                        trans_terms[term_id]['opinion_position'][pos_id][1] += \
                            offset

        return trans_words, trans_terms

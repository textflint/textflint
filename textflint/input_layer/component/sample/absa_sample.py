r"""
ABSASample Class
============================================

"""
from .sample import Sample
from ..field import TextField
from ....common.utils.error import FlintError

__all__ = ['ABSASample']


class ABSASample(Sample):
    r"""
    ABSASample Class

    """

    def __init__(
            self,
            data,
            trans_id=None,
            origin=None,
            sample_id=None
    ):
        self.sentence = None
        self.trans_id = trans_id

        super().__init__(data, origin=origin, sample_id=sample_id)

    def __repr__(self):
        return 'ABSASample'

    def check_data(self, data):
        r"""
        Check the format of input data.

        :param dict data: data name

        """
        assert 'sentence' in data and isinstance(data['sentence'], str), \
            "Sentence should be in data, and type of sentence should be str."
        assert 'term_list' in data and isinstance(data['term_list'], dict), \
            "Term_list should be in data, and type of term_list should be dict."

        for term_id, term_dict in data['term_list'].items():
            assert isinstance(term_id, str), \
                "Type of key of items in term_list should be str."
            assert isinstance(term_dict, dict), \
                "Type of value of items in term_list should be dict."
            assert 'polarity' in term_dict and \
                isinstance(term_dict['polarity'], str) \
                and len(term_dict['polarity']) > 0, \
                "Polarity should be in term_list, " \
                "type of polarity should be str, " \
                "and length of polarity should larger than 0."

            assert 'term' in term_dict and isinstance(term_dict['term'], str) \
                and len(term_dict['term']) > 0, \
                "Term should be in term_list, " \
                "type of term should be str and length of term " \
                "should larger than 0."
            assert 'from' in term_dict and isinstance(term_dict['from'], int) \
                and term_dict['from'] >= 0, \
                "From should be in term_list, and from should be " \
                "an int not less than 0."

            assert 'to' in term_dict and isinstance(term_dict['to'], int) \
                and term_dict['to'] >= 0, \
                "To should be in term_list, and to should be " \
                "an int not less than 0."
            assert 'opinion_words' in term_dict \
                and isinstance(term_dict['opinion_words'], list) \
                and len(term_dict['opinion_words']) > 0, \
                "Opinion_words should be in term_list, " \
                "and type of opinion_words should be list " \
                "and the length of opinion_words should larger than 0."

            for opinion_word in term_dict['opinion_words']:
                assert isinstance(opinion_word, str) \
                    and len(opinion_word) > 0, \
                    "The type of elements of opinion_words should be str " \
                    "and their length should larger than 0."
            assert 'opinion_position' in term_dict \
                and isinstance(term_dict['opinion_position'], list) \
                and len(term_dict['opinion_position']) > 0, \
                "Opinion_position should be in term_list, " \
                "type of opinion_position should be str, " \
                "and the length of opinion_position should larger than 0."

            for positions in term_dict['opinion_position']:
                assert isinstance(positions, list) and len(positions) == 2, \
                    "The type of elements of opinion_position should be list " \
                    "and its length should be 2."
                for position in positions:
                    assert position >= 0, \
                        "Opinion position should be not less than 0."
                assert positions[1] > positions[0], \
                    "The end position of opinion word should larger than " \
                    "its start position."

    def load(self, data):
        r"""
        Load the legal data and convert it into SASample.

        :param dict data: data name

        """
        self.data = data
        self.sentence = TextField(data['sentence'])
        self.term_list = data['term_list']
        self.terms = self.tokenize_term_list()

        if not self.is_legal():
            raise ValueError(
                "Term list {0} is not legal, aspect words or "
                "opinion words are in the wrong position."
                .format(self.term_list))

        self.contra = None
        self.multi = None
        self.id = None

        if 'contra' in data:
            self.contra = data['contra']
        if 'multi' in data:
            self.multi = data['multi']
        if 'id' in data:
            self.id = data['id']
        if self.trans_id is None and 'trans_id' in data:
            self.trans_id = data['trans_id']

    def dump(self):
        r"""
        Dump the legal data.

        :return dict: output of transformed data

        """
        self.update_term_list(self)

        if not self.is_legal():
            raise ValueError("Term list {0} is not legal, aspect words "
                             "or opinion words are in the wrong position."
                             .format(self.term_list))

        return {
            'sentence': self.sentence.text,
            'term_list': self.term_list,
            'contra': self.contra,
            'multi': self.multi,
            'id': self.id,
            'trans_id': self.trans_id,
            'sample_id': self.sample_id
        }

    def is_legal(self):
        r"""
        Check whether aspect words and opinion words are
            in the correct position.

        :return bool: whether format of data is legal.

        """

        terms = self.terms
        copy_sent = self.sentence.words

        for term_id in terms:
            opinions_spans = terms[term_id]['opinion_position']

            for i in range(len(opinions_spans)):
                position = opinions_spans[i]
                opinion_from = position[0]
                opinion_to = position[1]
                opinion = copy_sent[opinion_from:opinion_to]
                if opinion_from == opinion_to or opinion == '':
                    return False

            term_from = terms[term_id]['from']
            term_to = terms[term_id]['to']
            aspect = copy_sent[term_from:term_to]
            if term_from == term_to or aspect == '':
                return False

        return True

    def tokenize_term_list(self):
        r"""
        Tokenize the term list of ABSASample.

        :return list: terms in ABSASample

        """
        terms = {}
        copy_sent = self.sentence.text
        term_list = self.term_list

        for term_id in term_list:
            if term_id not in terms:
                terms[term_id] = {}
            opinion_words = []
            opinion_position = []
            opinions = term_list[term_id]['opinion_words']
            opinions_spans = term_list[term_id]['opinion_position']
            polarity = term_list[term_id]['polarity']

            for i in range(len(opinions)):
                position = opinions_spans[i]
                opinion_from = position[0]
                opinion_to = position[1]
                left = ABSASample.text_processor.tokenize(
                    copy_sent[:opinion_from].strip())
                opinion = ABSASample.text_processor.tokenize(
                    copy_sent[opinion_from:opinion_to].strip())
                opinion_words.append([' '.join(opinion)])
                opinion_position.append([len(left), len(left) + len(opinion)])

            term_from = term_list[term_id]['from']
            term_to = term_list[term_id]['to']
            left = ABSASample.text_processor.tokenize(
                copy_sent[:term_from].strip())
            aspect = ABSASample.text_processor.tokenize(
                copy_sent[term_from:term_to].strip())
            terms[term_id]['term'] = term_list[term_id]['term']
            terms[term_id]['from'] = len(left)
            terms[term_id]['to'] = len(left) + len(aspect)
            terms[term_id]['polarity'] = polarity
            terms[term_id]['opinion_words'] = opinion_words
            terms[term_id]['opinion_position'] = opinion_position

        return terms

    def update_sentence(self, trans_sentence):
        r"""
        Update the sentence of ABSASample.

        :param str|list trans_sentence: updated sentence

        """
        if not isinstance(trans_sentence, str or list):
            raise TypeError("Transformed sentence requires 'list' or 'str, "
                            "but got {0}".format(trans_sentence))
        if len(trans_sentence) == 0:
            raise ValueError("Length of transformed sentence "
                             "should be larger than 0, but got {0}"
                             .format(len(trans_sentence)))

        self.sentence = TextField(trans_sentence)

    def update_terms(self, trans_terms):
        r"""
        Update the terms of ABSASample.

        :param dict trans_terms: updated terms

        """
        self.terms = trans_terms

    def update_term_list(self, sample):
        r"""
        Update the term_list of ABSASample.

        :param ABSAsample sample: updated sample

        """
        if not sample.is_legal():
            raise FlintError("Term list {0} is not legal, "
                             "aspect words or opinion words are "
                             "in the wrong position."
                             .format(sample.terms))
        terms = sample.terms

        term_list = sample.term_list
        copy_sent = ABSASample.text_processor.tokenize(
            sample.sentence.text)
        trans_term_list = term_list

        for term_id in term_list:
            opinions_spans = terms[term_id]['opinion_position']

            for i in range(len(opinions_spans)):
                position = opinions_spans[i]
                opinion_from = position[0]
                opinion_to = position[1]
                left = ABSASample.text_processor.inverse_tokenize(
                    copy_sent[:opinion_from])
                if left != '':
                    left += ' '
                opinion = ABSASample.text_processor.inverse_tokenize(
                    copy_sent[opinion_from:opinion_to])
                terms[term_id]['opinion_words'][i] = opinion
                trans_term_list[term_id]['opinion_words'][i] = opinion
                trans_term_list[term_id]['opinion_position'][i] = \
                    [len(left), len(left) + len(opinion)]

            term_from = terms[term_id]['from']
            term_to = terms[term_id]['to']
            left = ABSASample.text_processor.inverse_tokenize(
                copy_sent[:term_from])
            if left != '':
                left += ' '
            aspect = ABSASample.text_processor.inverse_tokenize(
                copy_sent[term_from:term_to])
            terms[term_id]['term'] = aspect
            trans_term_list[term_id]['term'] = aspect
            trans_term_list[term_id]['id'] = term_id
            trans_term_list[term_id]['from'] = len(left)
            trans_term_list[term_id]['to'] = len(left) + len(aspect)
            trans_term_list[term_id]['polarity'] = terms[term_id]['polarity']

    def insert_field_before_indices(self, field, indices, items):
        r"""
        Insert items of multi given scopes before indices of field value
        at the same time.

        :param str field: transformed field
        :param list indices: indices of insert positions
        :param list items: insert items
        :return ~textflint.ABSAsample: modified sample

        """
        new_items = items
        sample = self.clone(self)

        for i, index in enumerate(indices):
            if isinstance(items[i], list):
                offset = len(items[i])
            else:
                offset = len(ABSASample.text_processor.tokenize(
                    items[i]))

            for term_id in sample.terms:
                if index == sample.terms[term_id]['from'] == \
                        sample.terms[term_id]['to']:
                    sample.terms[term_id]['to'] += offset
                elif index <= sample.terms[term_id]['from']:
                    sample.terms[term_id]['from'] += offset
                    sample.terms[term_id]['to'] += offset
                elif index < sample.terms[term_id]['to']:
                    sample.terms[term_id]['to'] += offset

                for opinion_word in sample.terms[term_id]['opinion_position']:
                    if index == opinion_word[0] == opinion_word[1]:
                        opinion_word[1] += offset
                    elif index <= opinion_word[0]:
                        opinion_word[0] += offset
                        opinion_word[1] += offset
                    elif index < opinion_word[1]:
                        opinion_word[1] += offset

            sample = super(
                ABSASample, sample).insert_field_before_indices(
                field, indices, new_items)
            sample.update_term_list(sample)

        return sample

    def insert_field_before_index(self, field, ins_index, new_item):
        r"""
        Insert items of multi given scope before index of field value.

        :param str field: transformed field
        :param int|list ins_index: index of insert position
        :param str|list new_item: insert item
        :return ~textflint.ABSAsample: modified sample

        """
        return self.insert_field_before_indices(field, [ins_index], [new_item])

    def insert_field_after_indices(self, field, indices, items):
        r"""
        Insert items of multi given scopes  after indices of
        field value at the same time.

        :param str field: transformed field
        :param list indices: indices of insert positions
        :param list items: insert items
        :return ABSAsample: modified sample

        """
        sample = self.clone(self)

        for i, index in enumerate(indices):
            if isinstance(items[i], list):
                offset = len(items[i])
            else:
                offset = len(ABSASample.text_processor.tokenize(
                    items[i]))

            for term_id in sample.terms:
                if index == sample.terms[term_id]['from'] - \
                        1 == sample.terms[term_id]['to'] - 1:
                    sample.terms[term_id]['to'] += offset
                elif index < sample.terms[term_id]['from']:
                    sample.terms[term_id]['from'] += offset
                    sample.terms[term_id]['to'] += offset
                elif index < sample.terms[term_id]['to'] - 1:
                    sample.terms[term_id]['to'] += offset
                for opinion_word in sample.terms[term_id]['opinion_position']:
                    if index == opinion_word[0] - 1 == opinion_word[1] - 1:
                        opinion_word[1] += offset
                    elif index < opinion_word[0]:
                        opinion_word[0] += offset
                        opinion_word[1] += offset
                    elif index < opinion_word[1] - 1:
                        opinion_word[1] += offset
            sample = super(
                ABSASample, sample).insert_field_after_indices(
                field, indices, items)
            sample.update_term_list(sample)

        return sample

    def insert_field_after_index(self, field, ins_index, new_item):
        r"""
        Insert items of multi given scope after index of field value.

        :param str field: transformed field
        :param int|list ins_index: index of insert position
        :param str|list new_item: insert item
        :return ~textflint.ABSAsample: modified sample

        """
        return self.insert_field_after_indices(field, [ins_index], [new_item])

    def delete_field_at_indices(self, field, indices):
        r""" Delete items of given scopes of field value.

        :param str field: transformed field
        :param list indices: indices of delete positions
        :return ABSAsample: modified sample

        """
        assert len(indices) > 0
        sample = self.clone(self)
        sample = super(
            ABSASample,
            sample).delete_field_at_indices(
            field,
            indices)

        sorted_indices = sorted(indices, key=lambda x: x, reverse=True)

        if not isinstance(sorted_indices[0], list):
            sorted_indices = [sorted_indices]

        for del_index in sorted_indices:
            del_from = del_index[0]
            if len(del_index) != 1:
                del_len = del_index[1] - del_index[0]
            else:
                del_len = 1
            for term_id in sample.terms:
                if del_from < sample.terms[term_id]['from']:
                    sample.terms[term_id]['from'] -= del_len
                    sample.terms[term_id]['to'] -= del_len
                elif del_from < sample.terms[term_id]['to']:
                    sample.terms[term_id]['to'] -= del_len
                for opinion_word in sample.terms[term_id]['opinion_position']:
                    if del_from < opinion_word[0]:
                        opinion_word[0] -= del_len
                        opinion_word[1] -= del_len
                    elif del_from < opinion_word[1]:
                        opinion_word[1] -= del_len
            sample.update_term_list(sample)

        return sample

    def delete_field_at_index(self, field, del_index):
        r""" Delete items of given scopes of field value.

        :param str field: transformed field
        :param list del_index: index of delete position
        :return ~textflint.ABSAsample: modified sample

        """
        return self.delete_field_at_indices(field, [del_index])

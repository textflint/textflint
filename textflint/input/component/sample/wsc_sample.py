r"""
WSCSample Class
============================================

"""
from copy import deepcopy
import nltk
import string

from .sample import Sample
from ..field import TextField
from ....common.utils.error import FlintError

__all__ = ['WSCSample']

from ....common.utils.list_op import normalize_scope


class WSCSample(Sample):
    r"""
    WSCSample Class

    """

    def __init__(
        self,
        data,
        origin=None,
        sample_id=None
    ):
        r"""
        WSC Sample class to hold the necessary info
        and provide atomic operations.

        :param json data:  THe json obj that contains data info.
        :param ~Sample origin: Original sample obj.
        :param int sample_id: sample index

        """
        super().__init__(data, origin=origin, sample_id=sample_id)
        self.data = data

    def __repr__(self):
        return 'WSCSample'

    def check_data(self, data):
        r"""
        Check whether 'text', 'target' and 'label' is legal

        :param dict data: contains 'text', 'target' and 'label' keys.
        :return:

        example:
        {
            "text":"The city councilmen refused the demonstrators a permit because they feared violence.",
            "target":{"span1_index":4, "span2_index":9,
                    "span1_text":"the demonstrators","span2_text":"they"},
            "label":false
            "index":2
        }
        """
        #print("data", data['text'])

        assert 'text' in data and isinstance(data['text'], str), \
            "text should be in data, and " \
            "the type of context should be str"
        assert 'index' in data and isinstance(data['index'], int), \
            "index should be in data, and " \
            "the type of context should be int"
        assert 'target' in data and isinstance(data['target'], dict), \
            "target should be in data, and the type of context should be dict"
        assert 'label' in data and isinstance(data['label'], int), \
            "label should be in data, and the type of context should be int"
        target = data["target"]
        assert 'noun1_idx' in target and isinstance(target['noun1_idx'], int), \
            "noun1_idx should be in target, " \
            "type of noun1_idx should be int "
        assert 'noun2_idx' in target and isinstance(target['noun2_idx'], int), \
            "noun2_idx should be in target, " \
            "type of noun2_idx should be int "
        assert 'pron_idx' in target and isinstance(target['pron_idx'], int), \
            "pron_idx should be in target, " \
            "type of pron_idx should be int "
        assert 'noun1' in target and isinstance(target['noun1'], str), \
            "noun1 should be in target, " \
            "type of noun1 should be str "
        assert 'noun2' in target and isinstance(target['noun1'], str), \
            "noun2 should be in target, " \
            "type of noun2 should be str "
        assert 'pron' in target and isinstance(target['pron'], str), \
            "pron should be in target, " \
            "type of pron should be str "

    def load(self, data):
        r"""
        Convert data dict which contains essential information to WSCSample.

        :param dict data: contains 'text', 'target' and 'label' keys.
        :return:

        """
        self.text = TextField(data['text'])
        self.target = data['target']
        self.label = data['label']
        self.index = data['index']

        self.noun1 = data["target"]["noun1"]
        self.noun2 = data["target"]["noun2"]
        self.noun1_idx = data["target"]["noun1_idx"]
        self.noun2_idx = data["target"]["noun2_idx"]
        self.pron = data["target"]["pron"]
        self.pron_idx = data["target"]["pron_idx"]

        if not self.is_legal():
            raise ValueError("noun index and pronoun index should be greater than or equal to zero")

    def dump(self):
        r"""
        Dump the legal data.

        :return dict: output of transformed data

        """
        if not self.is_legal():
            raise ValueError("noun and noun index do not match, "
                             "pronoun and pronoun index do not match"
                             .format(self.target))
        return {
            'text': self.text.text,
            'target': self.target,
            'label': self.label,
            'index': self.index
        }



    def is_legal(self):
        r"""
        Validate whether the sample is legal

        """


        if self.noun1_idx < 0:
            return False
        if self.noun2_idx < 0:
            return False
        if self.pron_idx < 0:
            return False

        return True

    def delete_field_at_index(self, field, del_index):
        """ Delete items of given scopes of field value.

        :param string field: transformed field
        :param list del_index: index of delete position
        :return WSDSample sample: a modified sample
        """
        return self.delete_field_at_indices(field, [del_index])

    def delete_field_at_indices(self, field, indices):
        r"""
        Delete items of given scopes of field value.

        :param str field: field name
        :param list indices: list of int/list/slice, modified scopes
        :return: modified Sample

        """
        assert len(indices) > 0
        # sample = self.clone(self)
        target = deepcopy(self.target)
        for index in indices:
            scope = normalize_scope(index)
            offset = scope[1] - scope[0]
            for word_idx in ['noun1_idx', 'noun2_idx', 'pron_idx']:
                if scope[1] < target[word_idx]:
                    target[word_idx] -= offset
        sample = super(WSCSample, self).delete_field_at_indices(field, indices)
        sample.target = target
        for key in target.keys():
            setattr(sample, key, target[key])
        return sample

    def insert_field_before_index(self, field, index, items):
        r"""
        Insert item before index of field value.

        :param str field: field name
        :param int index: modified scope
        :param items: inserted item
        :return: modified Sample

        """
        return self.insert_field_before_indices(field, [index], [items])

    def insert_field_before_indices(self, field, indices, items):
        r"""
        Insert items of multi given scopes before indices of
        field value at the same time.
        :param str field: field name
        :param list indices: list of int/list/slice, modified scopes
        :param list items: inserted items
        :return: modified Sample
        """
        target = deepcopy(self.target)

        for i, index in enumerate(indices):
            if isinstance(items[i], list):
                offset = len(items[i])
            else:
                items[i] = items[i].split(' ')
                offset = len(items[i])

            for word_idx in ['noun1_idx', 'noun2_idx', 'pron_idx']:
                if index <= target[word_idx]:
                    target[word_idx] += offset

        sample = super(WSCSample, self).insert_field_before_indices(
            field, indices, items)
        sample.target = target
        for key in target.keys():
            setattr(sample, key, target[key])
        return sample

    def insert_field_after_index(self, field, index, new_item):
        r"""
        Insert item after index of field value.

        :param str field: field name
        :param int index: modified scope
        :param new_item: inserted item
        :return: modified Sample

        """

        return self.insert_field_after_indices(field, [index], [new_item])

    def insert_field_after_indices(self, field, indices, items):
        r"""
        Insert items of multi given scopes after indices of
        field value at the same time.
        :param str field: field name
        :param list indices: list of int/list/slice, modified scopes
        :param list items: inserted items
        :return: modified Sample
        """
        target = deepcopy(self.target)

        for i, index in enumerate(indices):
            if isinstance(items[i], list):
                offset = len(items[i])
            else:
                items[i] = items[i].split(' ')
                offset = len(items[i])

            for word_idx in ['noun1_idx', 'noun2_idx', 'pron_idx']:
                if index < target[word_idx]:
                    target[word_idx] += offset

        sample = super(WSCSample, self).insert_field_after_indices(
            field, indices, items)
        sample.target = target
        for key in target.keys():
            setattr(sample, key, target[key])
        return sample

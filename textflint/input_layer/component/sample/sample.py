r"""
Base Sample Abstract Class
============================================

"""
import copy
from abc import ABC, abstractmethod

from ..field import ListField, TextField
from ....common.settings import MODIFIED_MASK
from ....common.preprocess.en_processor import EnProcessor
__all__ = ['Sample']


class Sample(ABC):
    r"""
    Base Sample class to hold the necessary info and provide atomic operations

    """

    text_processor = EnProcessor()

    def __init__(
        self,
        data,
        origin=None,
        sample_id=None
    ):
        r"""
        :param dict data: The dict obj that contains data info.
        :param ~textflint.sample origin: original sample obj.
        :param int sample_id: sampleindex

        """
        self.origin = origin if origin else self
        self.log = []
        self.check_data(data)
        self.load(data)
        self.sample_id = sample_id

    def __repr__(self):
        return 'Sample'

    def get_value(self, field):
        r"""
        Get field value by field_str.

        :param str field: field name
        :return: field value

        """
        return copy.deepcopy(getattr(self, field).field_value)

    def get_words(self, field):
        r"""
        Get tokenized words of given textfield

        :param str field: field name
        :return: tokenized words

        """
        field_obj = getattr(self, field)
        assert isinstance(field_obj, TextField), \
            f"{field} is not a text field, get words failed!"

        return field_obj.words[:]

    def get_text(self, field):
        r"""
        Get text string of given textfield

        :param str field: field name
        :return string: text

        """
        field_obj = getattr(self, field)
        assert isinstance(field_obj, TextField), \
            f"{field} is not a text field, get text failed!"

        return field_obj.text

    def get_mask(self, field):
        r"""
        Get word masks of given textfield

        :param str field: field name
        :return: list of mask values

        """
        field_obj = getattr(self, field)
        assert isinstance(field_obj, TextField), \
            f"{field} is not a text field, get mask failed!"

        return field_obj.mask[:]

    def get_sentences(self, field):
        r"""
        Get split sentences of given textfield

        :param str field: field name
        :return: list of sentences

        """
        field_obj = getattr(self, field)
        assert isinstance(field_obj, TextField), \
            f"{field} is not a text field, get sentences failed!"

        return field_obj.sentences[:]

    def get_pos(self, field):
        r"""
        Get text field pos tags.
        :param str field: field name
        :return: pos tag list

        """
        field_obj = getattr(self, field)
        assert isinstance(field_obj, TextField), \
            f"{field} is not a text field, get pos tags failed!"

        return field_obj.pos_tagging[:]

    def get_ner(self, field):
        r"""
        Get text field ner tags

        :param str field: field name
        :return: ner tag list

        """
        field_obj = getattr(self, field)
        assert isinstance(field_obj, TextField), \
            f"{field} is not a text field, get named entities failed!"

        return field_obj.ner[:]

    def replace_fields(self, fields, field_values, field_masks=None):
        r"""
        Fully replace multi fields at the same time and return new sample.
        Notice: Not suggest use this API as it will set mask values of TextField
        to MODIFIED_MASK.

        :param list fields: field str list
        :param list field_values: field value list
        :param list field_masks: indicate mask values, useful for printable text
        :return: Modified Sample

        """
        assert len(fields) == len(field_values), \
            f"Fields length {len(fields)} unequal with " \
            f"field values {len(field_values)}"
        if field_masks:
            assert len(fields) == len(field_masks), \
                f"Fields length {len(fields)} unequal " \
                f"with field masks {len(field_masks)}"

        sample = self.clone(self)

        for index, field in enumerate(fields):
            origin_field = getattr(sample, field)
            assert isinstance(field_values[index], origin_field.field_type), \
                f"Cant replace {field} with type {field_values[index]}"

            new_field = origin_field.new_field(field_values[index])
            if isinstance(origin_field, TextField):
                masks_values = field_masks[index] if field_masks \
                    else [MODIFIED_MASK] * len(new_field)
                new_field.replace_mask(masks_values)

            setattr(sample, field, new_field)

        return sample

    def replace_field(self, field, field_value, field_mask=None):
        r"""
        Fully replace single field and return new sample.
        Notice: Not suggest use this API as it will set mask values of
        TextField to MODIFIED_MASK.

        :param str field: field str
        :param field_value: field_type
        :param list field_mask: indicate mask value of field
        :return: Modified Sample

        """
        fields_mask = [field_mask] if field_mask else None

        return self.replace_fields([field], [field_value], fields_mask)

    def replace_field_at_indices(self, field, indices, items):
        r"""
        Replace items of multi given scopes of field value at the same time.
        Stay away from the complex function !!!

        Be careful of your input list shape.

        :param str field: field name
        :param list of int|list|slice indices:
            each index can be int indicate replace single item or their list
                like [1, 2, 3],
            can be list like (0,3) indicate replace items from
                0 to 3(not included),
            can be slice which would be convert to list.
        :param items:
        :return: Modified Sample

        """
        assert isinstance(field, str) & isinstance(indices, list) \
               & isinstance(items, list), \
            f"Unequal( field length {0}, indices length {1}, items length {2}"
        assert len(indices) == len(items)

        sample = self.clone(self)
        field_obj = getattr(self, field)
        assert isinstance(field_obj, (ListField, TextField))
        rep_field = field_obj.replace_at_indices(indices, items)
        setattr(sample, field, rep_field)

        return sample

    def replace_field_at_index(self, field, index, items):
        r"""
        Replace items of given scope of field value.

        Be careful of your input list shape.

        :param str field: field name
        :param int|list|slice index:
            can be int indicate replace single item or list like [1, 2, 3],
            can be list like (0,3) indicate replace items
                from 0 to 3(not included),
            can be slice which would be convert to list.
        :param str|list items: shape: indices_num, correspond to field_sub_items
        :return: Modified Sample

        """
        return self.replace_field_at_indices(field, [index], [items])

    def unequal_replace_field_at_indices(self, field, indices, rep_items):
        r"""
        Replace scope items of field value with rep_items which may
        not equal with scope.

        :param field: field str
        :param indices: list of int/tupe/list
        :param rep_items: list
        :return: Modified Sample

        """
        assert len(indices) == len(rep_items) > 0
        sample = self.clone(self)
        sorted_items, sorted_indices = zip(
            *sorted(zip(rep_items, indices), key=lambda x: x[1], reverse=True))

        for idx, sorted_token in enumerate(sorted_items):
            sample = sample.delete_field_at_index(field, sorted_indices[idx])
            insert_index = sorted_indices[idx] \
                if isinstance(sorted_indices[idx], int) \
                else sorted_indices[idx][0]
            field_obj = getattr(sample, field)
            if insert_index > len(field_obj):
                raise ValueError('Cant replace items at range {0}'
                                 .format(sorted_indices[idx]))
            elif insert_index == len(field_obj):
                sample = sample.insert_field_after_index(
                    field, insert_index - 1, sorted_token)
            else:
                sample = sample.insert_field_before_index(
                    field, insert_index, sorted_token)

        return sample

    def delete_field_at_indices(self, field, indices):
        r"""
        Delete items of given scopes of field value.

        :param str field: field name
        :param list of int|list|slice indices:
            shape：indices_num
            each index can be int indicate delete single item or their list
                like [1, 2, 3],
            can be list like (0,3) indicate replace items
                from 0 to 3(not included),
            can be slice which would be convert to list.
        :return: Modified Sample

        """
        sample = self.clone(self)
        field_obj = getattr(sample, field)

        assert isinstance(field_obj, (ListField, TextField))
        del_field = field_obj.delete_at_indices(indices)
        setattr(sample, field, del_field)

        return sample

    def delete_field_at_index(self, field, index):
        r"""
        Delete items of given scopes of field value.

        :param str field: field value
        :param int|list|slice index:
            can be int indicate delete single item or their list like [1, 2, 3],
            can be list like (0,3) indicate replace items
                from 0 to 3(not included),
            can be slice which would be convert to list.
        :return: Modified Sample

        """
        return self.delete_field_at_indices(field, [index])

    def insert_field_before_indices(self, field, indices, items):
        r"""
        Insert items of multi given scopes before indices of field value
        at the same time.

        Stay away from the complex function !!!
        Be careful of your input list shape.

        :param str field: field name
        :param indices: list of int, shape：indices_num, list like [1, 2, 3]
        :param items: list of str/list,
            shape: indices_num, correspond to indices
        :return: Modified Sample

        """
        sample = self.clone(self)

        field_obj = getattr(sample, field)
        assert isinstance(field_obj, (ListField, TextField))
        rep_obj = field_obj.insert_before_indices(indices, items)
        setattr(sample, field, rep_obj)

        return sample

    def insert_field_before_index(self, field, index, items):
        r"""
        Insert items of multi given scope before index of field value.

        :param str field: field name
        :param int index: indicate which index to insert items
        :param str|list items: items to insert
        :return: Modified Sample

        """
        return self.insert_field_before_indices(field, [index], [items])

    def insert_field_after_indices(self, field, indices, items):
        r"""
        Insert items of multi given scopes  after indices of field value
        at the same time.

        Stay away from the complex function !!!
        Be careful of your input list shape.

        :param str field: field name
        :param indices: list of int, shape：indices_num, like [1, 2, 3]
        :param items: list of str/list
            shape: indices_num, correspond to indices
        :return: Modified Sample

        """
        sample = self.clone(self)

        field_obj = getattr(sample, field)
        assert isinstance(field_obj, (ListField, TextField))
        rep_obj = field_obj.insert_after_indices(indices, items)
        setattr(sample, field, rep_obj)

        return sample

    def insert_field_after_index(self, field, index, items):
        r"""
        Insert items of multi given scope after index of field value

        :param str field: field name
        :param int index: indicate where to apply insert
        :param str|list items: shape: indices_num, correspond to field_sub_items
        :return: Modified Sample

        """
        return self.insert_field_after_indices(field, [index], [items])

    def swap_field_at_index(self, field, first_index, second_index):
        r"""
        Swap items between first_index and second_index of field value.

        :param str field: field name
        :param int first_index:
        :param int second_index:
        :return: Modified Sample

        """
        sample = self.clone(self)

        field_obj = getattr(sample, field)
        assert isinstance(field_obj, (ListField, TextField))
        rep_obj = field_obj.swap_at_index(first_index, second_index)
        setattr(sample, field, rep_obj)

        return sample

    @abstractmethod
    def check_data(self, data):
        r"""
        Check rare data format

        :param data: rare data input
        :return:

        """
        raise NotImplementedError

    @abstractmethod
    def load(self, data):
        r"""
        Parse data into sample field value.

        :param data: rare data input

        """
        raise NotImplementedError

    @abstractmethod
    def dump(self):
        r"""
        Convert sample info to input data json format.

        :return: dict object.

        """
        raise NotImplementedError

    @classmethod
    def clone(cls, original_sample):
        r"""
        Deep copy self to a new sample

        :param original_sample: sample to be copied
        :return: Sample instance

        """
        sample = copy.deepcopy(original_sample)
        sample.origin = original_sample.origin

        return sample

    @property
    def is_origin(self):
        r"""
        Return whether the sample is original Sample.

        """
        return self.origin is self

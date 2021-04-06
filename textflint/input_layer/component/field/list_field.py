"""
List Field Class
=====================

A helper class that represents input list values that to be modified.
"""

from .field import Field
from ....common.utils.list_op import *


class ListField(Field):
    r"""
    A helper class that represents input list values that to be modified.

    Operations which modify field_value would generate new Field instance.

    """
    def __init__(
        self,
        field_value,
        **kwargs
    ):
        """
        :param [str] field_value: The list that ListField represents.

        """
        if isinstance(field_value, str):
            field_value = list(field_value)
        super().__init__(field_value, field_type=list, **kwargs)

    def replace_at_indices(self, indices, new_items):
        r"""
        Replace items at indices.

        Notice: just support isometric replace.

        :param list[int|list|slice] indices:
            each index can be int indicate replace single item
            or their list like [1, 2, 3].
            each index can be list like (0,3) indicate replace items
            from 0 to 3(not included) or their list like [(0, 3), (5,6)]
            each index can be slice which would be convert to list.
        :param list new_items: items corresponding indices.
        :return: new field object.

        """
        field_value = replace_at_scopes(self.field_value, indices, new_items)

        return self.new_field(field_value)

    def replace_at_index(self, index, new_items):
        r"""
        Replace item at index.

        :param int|list|slice index:
            can be int indicate replace single item or their list like [1, 2, 3]
            can be list like (0,3) indicate replace items from 0
                to 3(not included) or their list like [(0, 3), (5,6)]
            can be slice which would be convert to list.
        :param list new_items: items corresponding index.
        :return: new field object.

        """
        return self.replace_at_indices([index], [new_items])

    def delete_at_indices(self, indices):
        r"""
        Delete items at indices.

        :param list[int|list|slice] indices:
            each index can be int indicate delete single item
            or their list like [1, 2, 3].
            each index can be list like (0,3) indicate replace items
            from 0 to 3(not included) or their list like [(0, 3), (5,6)]
            each index can be slice which would be convert to list.

        :return:  new field object.

        """
        field_value = delete_at_scopes(self.field_value, indices)

        return self.new_field(field_value)

    def delete_at_index(self, index):
        r"""
        Delete item at index.

        :param int|list|slice index:
            can be int indicate delete single item or their list like [1, 2, 3]
            can be list like (0,3) indicate replace items from 0
                to 3(not included) or their list like [(0, 3), (5,6)]
            can be slice which would be convert to list.
        :return: new field object.

        """
        return self.delete_at_indices([index])

    def insert_before_indices(self, indices, new_items):
        r"""
        Insert items before indices.

        :param list[int|list|slice] indices:
            each index can be int indicate insert single item
            or their list like [1, 2, 3].
            each index can be list like (0,3) indicate replace items
            from 0 to 3(not included) or their list like [(0, 3), (5,6)]
            each index can be slice which would be convert to list.
        :param list new_items: items corresponding indices.
        :return: new field object.

        """
        field_value = insert_before_indices(
            self.field_value, indices, new_items)

        return self.new_field(field_value)

    def insert_before_index(self, index, new_items):
        r"""
        Insert items before index.

        :param int|list|slice index:
            can be int indicate insert single item or their list like [1, 2, 3]
            can be list like (0,3) indicate replace items from 0
                to 3(not included) or their list like [(0, 3), (5,6)]
            can be slice which would be convert to list.
        :param list new_items: items corresponding index.
        :return: new field object.

        """
        return self.insert_before_indices([index], [new_items])

    def insert_after_indices(self, indices, new_items):
        r"""
         Insert item after index.

        :param list[int|list|slice] indices:
            each index can be int indicate insert single item
            or their list like [1, 2, 3].
            each index can be list like (0,3) indicate replace items
            from 0 to 3(not included) or their list like [(0, 3), (5,6)]
            each index can be slice which would be convert to list.
        :param list new_items: items corresponding indices.
        :return: new field object.

        """
        field_value = insert_after_indices(
            self.field_value, indices, new_items)

        return self.new_field(field_value)

    def insert_after_index(self, index, new_items):
        r"""
        Insert item after index.

        :param int|list|slice index:
            can be int indicate insert single item or their list like [1, 2, 3]
            can be list like (0,3) indicate replace items from 0
                to 3(not included) or their list like [(0, 3), (5,6)]
            can be slice which would be convert to list.
        :param list new_items: items corresponding index
        :return: new field object.

        """
        return self.insert_after_indices([index], [new_items])

    def swap_at_index(self, first_index, second_index):
        r"""
        Swap item between first_index and second_index.

        :param int first_index: index of first item
        :param int second_index: index of second item
        :return: new field object.

        """
        field_value = swap_at_index(
            self.field_value, first_index, second_index)

        return self.new_field(field_value)

    def __len__(self):
        return len(self.field_value)

    def __getitem__(self, key):
        return self.field_value[key]

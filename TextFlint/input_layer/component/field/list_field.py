"""
List Field Class
=====================

A helper class that represents input list values that to be modified.
"""

from .field import Field
from ....common.utils.list_op import *


class ListField(Field):
    """A helper class that represents input list values that to be modified.

    Operations which modify field_value would generate new Field instance.

    Attributes:
        field_value: list of str
            The list that ListField represents.
    """

    def __init__(self, field_value, **kwargs):
        if isinstance(field_value, str):
            field_value = list(field_value)
        # TODO, support empty list
        super().__init__(field_value, field_type=list, **kwargs)

    def replace_at_indices(self, indices, new_items):
        """ Replace items at indices.

        Notice: just support isometric replace.

        Args:
            indices: list of int/list/slice
                each index can be int indicate replace single item or their list like [1, 2, 3].
                each index can be list like (0,3) indicate replace items from 0 to 3(not included)
                    or their list like [(0, 3), (5,6)]
                each index can be slice which would be convert to list.
            new_items: list
                items corresponding indices.

        Returns:
            new field object.

        """
        field_value = replace_at_scopes(self.field_value, indices, new_items)

        return self.new_field(field_value)

    def replace_at_index(self, index, new_items):
        """ Replace item at index.

         Args:
             index: int/list/slice
                can be int indicate replace single item or their list like [1, 2, 3].
                can be list like (0,3) indicate replace items from 0 to 3(not included)
                    or their list like [(0, 3), (5,6)]
                can be slice which would be convert to list.
            new_items: list
                items corresponding index.

        Returns:
            new field object.

         """
        return self.replace_at_indices([index], [new_items])

    def delete_at_indices(self, indices):
        """ Delete items at indices.

        Args:
            indices: list of int/list/slice
                each index can be int indicate replace single item or their list like [1, 2, 3].
                each index can be list like (0,3) indicate replace items from 0 to 3(not included)
                    or their list like [(0, 3), (5,6)]
                each index can be slice which would be convert to list.

        Returns:
            new field object.

        """
        field_value = delete_at_scopes(self.field_value, indices)

        return self.new_field(field_value)

    def delete_at_index(self, index):
        """ Delete item at index.

        Args:
            index: int/list/slice
                can be int indicate replace single item or their list like [1, 2, 3].
                can be list like (0,3) indicate replace items from 0 to 3(not included)
                    or their list like [(0, 3), (5,6)]
                can be slice which would be convert to list.

        Returns:
            new field object.
        """

        return self.delete_at_indices([index])

    def insert_before_indices(self, indices, new_items):
        """ Insert items before indices.

        Args:
            indices: list of int
            new_items: list

        Returns:
            new field object.

        """
        field_value = insert_before_indices(
            self.field_value, indices, new_items)

        return self.new_field(field_value)

    def insert_before_index(self, index, new_items):
        """ Insert items before index.

        Args:
            index: int
            new_items: list
                items corresponding index.

        Returns:
            new field object.
        """

        return self.insert_before_indices([index], [new_items])

    def insert_after_indices(self, indices, new_items):
        """ Insert item after index.

        Args:
            indices: list of int
            new_items: list

        Returns:
            new field object.

        """
        field_value = insert_after_indices(
            self.field_value, indices, new_items)

        return self.new_field(field_value)

    def insert_after_index(self, index, new_items):
        """ Insert item after index.

         Args:
            index: int
            new_items: list

        Returns:
            new field object.

        """

        return self.insert_after_indices([index], [new_items])

    def swap_at_index(self, first_index, second_index):
        """ Swap item between first_index and second_index.

        Args:
            first_index: int
            second_index: int

        Returns:
            new field object.
        """

        field_value = swap_at_index(
            self.field_value, first_index, second_index)

        return self.new_field(field_value)

    def __len__(self):
        return len(self.field_value)

    def __getitem__(self, key):
        return self.field_value[key]


if __name__ == "__main__":
    x = ListField(['H', 'e', 'l', 'l', 'o', ',', ' ', 'h', 'a',
                   'p', 'p', 'y', ' ', 'w', 'o', 'l', 'r', 'd', '!'])

    print('--------------------test operations-------------------------')
    # test insert
    print('_'.join(x.field_value))
    insert_before = x.insert_before_index(
        0, ['H', 'e', 'l', 'l', 'o', ',', ' '])
    print('_'.join(insert_before.field_value))
    insert_after = x.insert_after_index(len(x) - 1, ['B', 'y', 'e', '!'])
    print('_'.join(insert_after.field_value))
    insert_after = x.insert_before_indices(
        [len(x) - 1, 1], [['B', 'y', 'e', '!'], ['xx', 'yy']])
    print('_'.join(insert_after.field_value))

    # test swap
    swap = x.swap_at_index(0, 1)
    print('_'.join(x.field_value))
    print('_'.join(swap.field_value))

    # test delete
    delete = x.delete_at_index(0)
    print('_'.join(x.field_value))
    print('_'.join(delete.field_value))
    delete = x.delete_at_indices([0, (3, 5), 2])
    print('_'.join(delete.field_value))

    # test replace
    replace = x.replace_at_index(0, '$')
    print('_'.join(replace.field_value))
    replace = x.replace_at_indices(
        [[4, 6], [0, 3]], [['w', 'x'], ('$', "%", "&")])
    print('_'.join(replace.field_value))

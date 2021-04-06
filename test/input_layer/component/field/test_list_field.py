import unittest

from textflint.input_layer.component.field.list_field import *


class TestListField(unittest.TestCase):
    def test_list_field(self):
        x = ListField(['H', 'e', 'l', 'l', 'o', ',', ' ', 'h', 'a', 'p',
                       'p', 'y', ' ', 'w', 'o', 'l', 'r', 'd', '!'])

        # test insert
        insert_before = x.insert_before_index(
            0, ['H', 'e', 'l', 'l', 'o', ',', ' '])
        self.assertEqual(
            ['H', 'e', 'l', 'l', 'o', ',', ' ', 'H', 'e', 'l', 'l', 'o', ',',
             ' ', 'h', 'a', 'p', 'p', 'y', ' ', 'w', 'o', 'l', 'r', 'd', '!'],
            insert_before.field_value)
        insert_after = x.insert_after_index(len(x) - 1, ['B', 'y', 'e', '!'])
        self.assertEqual(
            ['H', 'e', 'l', 'l', 'o', ',', ' ', 'h', 'a', 'p', 'p', 'y', ' ',
             'w', 'o', 'l', 'r', 'd', '!', 'B', 'y', 'e', '!'],
            insert_after.field_value)
        insert_after = x.insert_before_indices(
            [len(x) - 1, 1], [['B', 'y', 'e', '!'], ['xx', 'yy']])
        self.assertEqual(
            ['H', 'xx', 'yy', 'e', 'l', 'l', 'o', ',', ' ', 'h', 'a', 'p', 'p',
             'y', ' ', 'w', 'o', 'l', 'r', 'd', 'B', 'y', 'e', '!', '!'],
            insert_after.field_value)

        # test swap
        swap = x.swap_at_index(0, 1)
        self.assertEqual(
            ['e', 'H', 'l', 'l', 'o', ',', ' ', 'h', 'a', 'p', 'p', 'y', ' ',
             'w', 'o', 'l', 'r', 'd', '!'], swap.field_value)

        # test delete
        delete = x.delete_at_index(0)
        self.assertEqual(
            ['e', 'l', 'l', 'o', ',', ' ', 'h', 'a', 'p', 'p', 'y', ' ', 'w',
             'o', 'l', 'r', 'd', '!'], delete.field_value)
        delete = x.delete_at_indices([0, (3, 5), 2])
        self.assertEqual(
            ['e', ',', ' ', 'h', 'a', 'p', 'p', 'y', ' ', 'w', 'o', 'l',
             'r','d', '!'], delete.field_value)

        # test replace
        replace = x.replace_at_index(0, '$')
        self.assertEqual(
            ['$', 'e', 'l', 'l', 'o', ',', ' ', 'h', 'a', 'p', 'p', 'y', ' ',
             'w', 'o', 'l', 'r', 'd', '!'], replace.field_value)
        replace = x.replace_at_indices(
            [[4, 6], [0, 3]], [['w', 'x'], ('$', "%", "&")])
        self.assertEqual(
            ['$', '%', '&', 'l', 'w', 'x', ' ', 'h', 'a', 'p', 'p', 'y', ' ',
             'w', 'o', 'l', 'r', 'd', '!'], replace.field_value)


if __name__ == "__main__":
    unittest.main()

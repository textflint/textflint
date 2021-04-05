import unittest

from TextFlint.input_layer.component.field.text_field import *


class TestTextField(unittest.TestCase):
    def test_text_field(self):
        self.assertRaises(ValueError, TextField, {})

        # test mask
        test_field = TextField(['A', 'man', 'goes', 'to', 'school'])
        test_field.set_mask(0, 1)
        self.assertEqual([1, 0, 0, 0, 0], test_field.mask)
        self.assertRaises(ValueError, test_field.set_mask, 0, 10)
        self.assertRaises(ValueError, test_field.set_mask, 10, 0)

        # test set_all_mask
        test_field.replace_mask([1]*len(test_field))
        self.assertEqual([1, 1, 1, 1, 1], test_field.mask)
        self.assertRaises(ValueError, test_field.replace_mask, 10)

        x = TextField("Fudan University natural language processing group. "
                      "Shanghai yangpu area.")
        # test field_value words sentence
        self.assertEqual(
            ('Fudan University natural language processing group. '
             'Shanghai yangpu area.',
             ['Fudan', 'University', 'natural', 'language', 'processing',
              'group', '.', 'Shanghai', 'yangpu', 'area', '.'],
             ['Fudan University natural language processing group.',
              'Shanghai yangpu area.']), (x.field_value, x.words, x.sentences))

        # test pos tag only the return format is correct, not the label
        self.assertEqual(len(x.words), len(x.pos_tagging))
        for tag in x.pos_tagging:
            self.assertTrue(isinstance(tag, str))

        # test ner only the return format is correct, not the label
        for entity in x.ner:
            self.assertTrue([type(i) for i in entity] == [str, int, int, str]
                            and entity[1] < entity[2])

        # test dp only the return format is correct, not the label
        self.assertEqual(len(x.words), len(x.dependency_parsing))
        for dp in x.dependency_parsing:
            self.assertTrue([type(i) for i in dp] == [str, str, int, str])

        # test operations
        # test insert
        insert_before = x.insert_before_index(0, ['test '])
        self.assertEqual(
           ['test ', 'Fudan', 'University', 'natural', 'language', 'processing',
            'group', '.', 'Shanghai', 'yangpu', 'area', '.'],
            insert_before.field_value)
        insert_before = x.insert_before_indices(
            [0, 2, 4], ['Wang', 'Xiao', ['fdu', 'jiangwan', 'cross_2']])
        self.assertEqual(
            ['Wang', 'Fudan', 'University', 'Xiao', 'natural', 'language',
             'fdu','jiangwan', 'cross_2', 'processing', 'group', '.',
             'Shanghai', 'yangpu', 'area', '.'], insert_before.field_value)

        insert_after = x.insert_after_index(len(x.words) - 1, [' haha', 'test'])
        self.assertEqual(
            ['Fudan', 'University', 'natural', 'language', 'processing',
             'group', '.','Shanghai', 'yangpu', 'area', '.', ' haha', 'test'],
            insert_after.field_value)
        insert_after = x.insert_after_indices(
            [0, 2, 7], ['Wang', 'Xiao', ['fdu', 'jiangwan', 'cross_2']])
        self.assertEqual(
            ['Fudan', 'Wang', 'University', 'natural', 'Xiao', 'language',
             'processing', 'group', '.', 'Shanghai', 'fdu', 'jiangwan',
             'cross_2', 'yangpu', 'area', '.'], insert_after.field_value)

        # test swap
        swap = x.swap_at_index(0, 1)
        self.assertEqual(['University', 'Fudan', 'natural', 'language',
                          'processing', 'group', '.', 'Shanghai', 'yangpu',
                          'area', '.'], swap.field_value)

        # test delete
        delete = x.delete_at_index(0)
        self.assertEqual(
            ['University', 'natural', 'language', 'processing', 'group', '.',
             'Shanghai', 'yangpu', 'area', '.'], delete.field_value)
        delete = x.delete_at_indices([0, [2, 4], len(x.words) - 1])
        self.assertEqual(['University', 'processing', 'group', '.', 'Shanghai',
                          'yangpu', 'area'], delete.field_value)

        # test replace
        replace = x.replace_at_index(0, '$')
        self.assertEqual(
            ['$', 'University', 'natural', 'language', 'processing',
             'group', '.', 'Shanghai', 'yangpu', 'area', '.'],
            replace.field_value)
        replace = x.replace_at_indices(
            [0, [2, 4], [7, 8]], ['$', ['wang', 'xiao'], 'fDu'])
        self.assertEqual(
            ['$', 'University', 'wang', 'xiao', 'processing', 'group', '.',
             'fDu', 'yangpu', 'area', '.'], replace.field_value)


if __name__ == "__main__":
    unittest.main()

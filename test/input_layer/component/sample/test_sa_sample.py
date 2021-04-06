import unittest

from textflint.input_layer.component.sample.sa_sample import *

data = {'x': "Brilliant and moving performances by "
             "Tom Courtenay and Peter Finch",
        'y': 'negative'}
sa_sample = SASample(data)


class TestSASample(unittest.TestCase):
    def test_load_sample(self):
        # test wrong data
        self.assertRaises(AssertionError, SASample, {'x': 'movie'})
        self.assertRaises(AssertionError, SASample, {'y': 'movie'})
        self.assertRaises(AssertionError, SASample, {'x': ''})
        self.assertRaises(ValueError, SASample, {'x': 'movie', 'y': 5})
        self.assertRaises(AssertionError, SASample, {'x': ['movie'],
                                                     'y': 'negative'})
        self.assertRaises(AssertionError, SASample, {'x': [], 'y': []})
        self.assertRaises(ValueError, SASample, {'x': 'the US', 'y': []})
    
    def test_insert_field_after_index(self):
        # test insert before index and mask
        ins_aft = sa_sample.insert_field_after_index('x', 2, '$$$')
        self.assertEqual('Brilliant and moving $$$ performances by '
                         'Tom Courtenay and Peter Finch',
                         ins_aft.get_text('x'))
        self.assertEqual(ins_aft.y.field_value, 'negative')
        self.assertEqual([0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
                         ins_aft.get_mask('x'))

    def test_unequal_replace_field_at_indices(self):
        # test insert before index and mask
        ins_aft = sa_sample.unequal_replace_field_at_indices(
            'x', [2], [['$$$', '123']])
        self.assertEqual('Brilliant and $$$ 123 performances by '
                         'Tom Courtenay and Peter Finch',
                         ins_aft.get_text('x'))
        self.assertEqual(ins_aft.y.field_value, 'negative')
        self.assertEqual([0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0],
                         ins_aft.get_mask('x'))

    def test_concat_token(self):
        # test wrong data
        self.assertRaises(AssertionError, sa_sample.concat_token, [])
        self.assertRaises(AssertionError, sa_sample.concat_token, 0)
        concat_tokens = sa_sample.concat_token(5)
        # test output
        for concat_token in concat_tokens:
            self.assertTrue(isinstance(concat_token, dict))
            self.assertEqual(
                concat_token['string'],
                ' '.join(sa_sample.get_words('x')[concat_token['indices'][0]:concat_token['indices'][1]]))

    def test_mask(self):
        # test does the mask label work
        new = sa_sample.insert_field_after_index('x', 0, 'abc')
        new = new.delete_field_at_index('x', 1)
        # TODO wait repair bug
        print(new.dump())
        print(new.get_mask('x'))

    def test_get_words(self):
        # test get words
        print(sa_sample.get_words('x'))
        self.assertEqual(
            ['Brilliant', 'and', 'moving', 'performances', 'by', 'Tom',
             'Courtenay', 'and', 'Peter', 'Finch'], sa_sample.get_words('x'))
        self.assertRaises(AssertionError, sa_sample.get_words, 'y')

    def test_get_text(self):
        # test get text
        self.assertEqual('Brilliant and moving performances by Tom Courtenay '
                         'and Peter Finch', sa_sample.get_text('x'))
        self.assertRaises(AssertionError, sa_sample.get_text, 'y')

    def test_get_value(self):
        # test get value
        self.assertEqual('Brilliant and moving performances by Tom Courtenay '
                         'and Peter Finch', sa_sample.get_value('x'))
        self.assertEqual('negative', sa_sample.get_value('y'))

    def test_dump(self):
        # test dump
        self.assertEqual({'x': 'Brilliant and moving performances by'
                               ' Tom Courtenay and Peter Finch',
                          'y': 'negative', 'sample_id': None}, sa_sample.dump())


if __name__ == "__main__":
    unittest.main()

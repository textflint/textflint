import unittest

from TextFlint.input_layer.component.sample.pos_sample import *

data = {'x': ['That', 'is', 'a', 'good', 'survey'],
        'y': ['DT', 'VBZ', 'DT', 'JJ', 'NN']}

pos_sample = POSSample(data)


class TestPOSSample(unittest.TestCase):
    def test_load_sample(self):
        # test wrong data
        self.assertRaises(AssertionError, POSSample, {'x': 'apple'})
        self.assertRaises(AssertionError, POSSample, {'y': 'apple'})
        self.assertRaises(AssertionError, POSSample, {'x': ''})
        self.assertRaises(ValueError, POSSample, {'x': [], 'y': []})
        self.assertRaises(ValueError, POSSample, {'x': ['That', 'is'],
                                                  'y': ['DT']})

    def test_insert_field_before_index(self):
        # test insert before index and mask
        ins_bef = pos_sample.insert_field_before_index('x', 0, '$$$')
        self.assertEqual(['$$$', 'That', 'is', 'a', 'good', 'survey'],
                         ins_bef.dump()['x'])
        self.assertEqual(ins_bef.dump()['y'], ['UNK'] + pos_sample.dump()['y'])
        self.assertEqual([2, 0, 0, 0, 0, 0], ins_bef.x.mask)
        self.assertRaises(IndexError, pos_sample.insert_field_before_index,
                          'x', 10, '$$$')
        self.assertRaises(ValueError, pos_sample.insert_field_before_index,
                          'x', -1, '$$$')

    def test_insert_field_after_index(self):
        # test insert before index and mask
        ins_aft = pos_sample.insert_field_after_index('x', 2, '$$$')
        self.assertEqual(['That', 'is', 'a', '$$$', 'good', 'survey'],
                         ins_aft.dump()['x'])
        self.assertEqual(ins_aft.dump()['y'], ['DT', 'VBZ', 'DT', 'UNK',
                                               'JJ', 'NN'])
        self.assertEqual([0, 0, 0, 2, 0, 0], ins_aft.x.mask)
        self.assertRaises(IndexError, pos_sample.insert_field_after_index,
                          'x', 10, '$$$')

    def test_delete_field_at_index(self):
        # test insert before index and mask
        del_sample = pos_sample.delete_field_at_index('x', [1, 3])
        self.assertEqual(['That', 'good', 'survey'], del_sample.dump()['x'])
        self.assertEqual(del_sample.dump()['y'], ['DT', 'JJ', 'NN'])
        self.assertEqual([0, 0, 0], del_sample.x.mask)
        self.assertRaises(ValueError, pos_sample.delete_field_at_index,
                          'x', [5, 4])

    def test_mask(self):
        # test does the mask label work
        new = pos_sample.insert_field_after_index('x', 0, 'abc')
        new = new.delete_field_at_index('x', 1)
        # TODO wait repair bug
        print(new.dump())
        print(new.x.mask)

    def test_get_words(self):
        # test get words
        self.assertEqual(['That', 'is', 'a', 'good', 'survey'],
                         pos_sample.get_words('x'))
        self.assertRaises(AssertionError, pos_sample.get_words, 'y')

    def test_get_text(self):
        # test get text
        self.assertEqual('That is a good survey', pos_sample.get_text('x'))
        self.assertRaises(AssertionError, pos_sample.get_text, 'y')

    def test_get_value(self):
        # test get value
        self.assertEqual(['That', 'is', 'a', 'good', 'survey'],
                         pos_sample.get_value('x'))
        self.assertEqual(['DT', 'VBZ', 'DT', 'JJ', 'NN'],
                         pos_sample.get_value('y'))

    def test_dump(self):
        # test dump
        self.assertEqual({'x': ['That', 'is', 'a', 'good', 'survey'],
                          'y': ['DT', 'VBZ', 'DT', 'JJ', 'NN'],
                          'sample_id': None},
                         pos_sample.dump())


if __name__ == "__main__":
    unittest.main()

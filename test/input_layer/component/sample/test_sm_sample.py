import unittest

from textflint.input_layer.component.sample.sm_sample import *

data = {'sentence1': 'MR zhang has 10 students',
        'sentence2': 'Mr zhang has 20 students',
        'y': '0'}
sm_sample = SMSample(data)


class TestSMSample(unittest.TestCase):
    def test_load_sample(self):
        # test wrong data
        self.assertRaises(AssertionError, SMSample,
                          {'sentence1': 'MR zhang has 10 students .'})
        self.assertRaises(AssertionError, SMSample,
                          {'sentence2': 'Mr zhang has 20 students .'})
        self.assertRaises(AssertionError, SMSample,
                          {'x': 'MR zhang has 10 students .'})
        self.assertRaises(AssertionError, SMSample, {'y': 'contradiction'})
        self.assertRaises(AssertionError, SMSample, {'sentence1': 11,
                                                     'sentence2': 22, 'y': '1'})
        self.assertRaises(AssertionError, SMSample, {'sentence1': [],
                                                     'sentence2': [], 'y': '0'})
        self.assertRaises(AssertionError, SMSample, {'sentence1': '',
                                                     'sentence2': ''})
        self.assertRaises(ValueError, SMSample, {
            'sentence1': 'MR zhang has 10 students .',
            'sentence2': 'Mr zhang has 20 students .', 'y': 1})

    def test_insert_field_after_index(self):
        # test insert before index and mask
        ins_aft = sm_sample.insert_field_after_index('sentence1', 2, '$$$')
        self.assertEqual('MR zhang has $$$ 10 students',
                         ins_aft.get_text('sentence1'))
        self.assertEqual(ins_aft.y.field_value, '0')
        self.assertEqual([0, 0, 0, 2, 0, 0], ins_aft.get_mask('sentence1'))

    def test_unequal_replace_field_at_indices(self):
        # test insert before index and mask
        ins_aft = sm_sample.unequal_replace_field_at_indices(
            'sentence2', [2], [['$$$', '123']])
        self.assertEqual('Mr zhang $$$ 123 20 students',
                         ins_aft.get_text('sentence2'))
        self.assertEqual(ins_aft.y.field_value, '0')
        self.assertEqual([0, 0, 2, 2, 0, 0], ins_aft.get_mask('sentence2'))


    def test_mask(self):
        # test does the mask label work
        new = sm_sample.insert_field_after_index('sentence1', 0, 'abc')
        new = new.delete_field_at_index('sentence1', 1)
        # TODO wait repair bug
        print(new.dump())
        print(new.get_mask('sentence1'))

    def test_get_words(self):
        # test get words
        print(sm_sample.get_words('sentence2'))
        self.assertEqual(['Mr', 'zhang', 'has', '20', 'students'],
                         sm_sample.get_words('sentence2'))
        self.assertRaises(AssertionError, sm_sample.get_words, 'y')

    def test_get_text(self):
        # test get text
        self.assertEqual('MR zhang has 10 students',
                         sm_sample.get_text('sentence1'))
        self.assertRaises(AssertionError, sm_sample.get_text, 'y')

    def test_get_value(self):
        # test get value
        self.assertEqual('MR zhang has 10 students',
                         sm_sample.get_value('sentence1'))
        self.assertEqual('0', sm_sample.get_value('y'))

    def test_dump(self):
        # test dump
        self.assertEqual({'sentence1': 'MR zhang has 10 students',
                          'sentence2': 'Mr zhang has 20 students',
                          'y': '0',
                          'sample_id': None}, sm_sample.dump())


if __name__ == "__main__":
    unittest.main()

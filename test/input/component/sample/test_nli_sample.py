import unittest

from textflint.input.component.sample.nli_sample import *

data = {'hypothesis': 'MR zhang has 10 students',
        'premise': 'Mr zhang has 20 students',
        'y': 'contradiction'}
nli_sample = NLISample(data)


class TestNLISample(unittest.TestCase):
    def test_load_sample(self):
        # test wrong data
        self.assertRaises(AssertionError, NLISample,
                          {'hypothesis': 'MR zhang has 10 students .'})
        self.assertRaises(AssertionError, NLISample,
                          {'premise': 'Mr zhang has 20 students .'})
        self.assertRaises(AssertionError, NLISample,
                          {'x': 'MR zhang has 10 students .'})
        self.assertRaises(AssertionError, NLISample, {'y': 'contradiction'})
        self.assertRaises(AssertionError, NLISample,
                          {'hypothesis': 11, 'premise': 22, 'y': 'contradiciton'})
        self.assertRaises(AssertionError, NLISample,
                          {'hypothesis': '', 'premise': ''})
        self.assertRaises(AssertionError, NLISample,
                          {'hypothesis': [], 'premise': 'haha', 'y': 'contradiciton'})
        self.assertRaises(ValueError, NLISample,
                          {'hypothesis': 'MR zhang has 10 students .',
                           'premise': 'Mr zhang has 20 students .', 'y': 1})

    def test_insert_field_after_index(self):
        # test insert before index and mask
        ins_aft = nli_sample.insert_field_after_index('hypothesis', 2, '$$$')
        self.assertEqual('MR zhang has $$$ 10 students',
                         ins_aft.get_text('hypothesis'))
        self.assertEqual(ins_aft.y.field_value, 'contradiction')
        self.assertEqual([0, 0, 0, 2, 0, 0], ins_aft.get_mask('hypothesis'))

    def test_unequal_replace_field_at_indices(self):
        # test insert before index and mask
        ins_aft = nli_sample.unequal_replace_field_at_indices('premise', [2],
                                                              [['$$$', '123']])
        self.assertEqual('Mr zhang $$$ 123 20 students',
                         ins_aft.get_text('premise'))
        self.assertEqual(ins_aft.y.field_value, 'contradiction')
        self.assertEqual([0, 0, 2, 2, 0, 0], ins_aft.get_mask('premise'))


    def test_mask(self):
        # test does the mask label work
        new = nli_sample.insert_field_after_index('hypothesis', 0, 'abc')
        new = new.delete_field_at_index('hypothesis', 1)
        # TODO wait repair bug
        print(new.dump())
        print(new.get_mask('hypothesis'))

    def test_get_words(self):
        # test get words
        print(nli_sample.get_words('premise'))
        self.assertEqual(['Mr', 'zhang', 'has', '20', 'students'],
                         nli_sample.get_words('premise'))
        self.assertRaises(AssertionError, nli_sample.get_words, 'y')

    def test_get_text(self):
        # test get text
        self.assertEqual('MR zhang has 10 students',
                         nli_sample.get_text('hypothesis'))
        self.assertRaises(AssertionError, nli_sample.get_text, 'y')

    def test_get_value(self):
        # test get value
        self.assertEqual('MR zhang has 10 students',
                         nli_sample.get_value('hypothesis'))
        self.assertEqual('contradiction', nli_sample.get_value('y'))

    def test_dump(self):
        # test dump
        self.assertEqual({'hypothesis': 'MR zhang has 10 students',
                          'premise': 'Mr zhang has 20 students',
                          'y': 'contradiction',
                          'sample_id': None}, nli_sample.dump())


if __name__ == "__main__":
    unittest.main()

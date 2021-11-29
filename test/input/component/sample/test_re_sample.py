import unittest

from textflint.input.component.sample.re_sample import RESample
from textflint.common.preprocess.en_processor import EnProcessor

data = {'x': ["``", "The", "situation", "is", "very", "serious", ",", "''",
              "Mattis", ",", "30", ",", "told", "reporters", "after",
              "meeting", "with", "Ban", "in", "New", "York", "."],
        'subj': [8, 8], 'obj': [10, 10], 'y': 'age', 'sample_id': None}
re_sample = RESample(data)


class TestRESample(unittest.TestCase):
    def test_load_sample(self):
        # test wrong data
        self.assertRaises(AssertionError, RESample,
                          {'x': ['The', 'situation']})
        self.assertRaises(
            AssertionError, RESample,
            {'x': ['The', 'situation'], 'subj':[0, 0], 'obj':[0, 0]})
        self.assertRaises(AssertionError, RESample,
                          {'x': ['The', 'situation'], 'y': 'None'})
        self.assertRaises(AssertionError, RESample, {'x': ''})
        self.assertRaises(
            ValueError, RESample,
            {'x': ['The', 'situation'], 'subj':[2, 2], 'obj':[0, 0], 'y': 'None'})
        self.assertRaises(
            ValueError, RESample,
            {'x': ['The', 'situation'],
             'subj': [1, 0], 'obj': [0, 0], 'y': 'None'})
        self.assertRaises(
            ValueError, RESample,
            {'x': ['The', 'situation'], 'subj': [], 'obj': [1, 1], 'y': 'None'})
        self.assertRaises(
            AssertionError, RESample,
            {'x': ['The', 'situation'], 'subj': [0, 0], 'obj': [1, 1], 'y': []})

    def test_get_words(self):
        # test get words
        self.assertEqual(data['x'], re_sample.get_words('x'))
        self.assertRaises(AssertionError, re_sample.get_words, 'y')

    def test_get_text(self):
        # test get text
        self.assertEqual(EnProcessor.inverse_tokenize(data['x']),
                         re_sample.get_text('x'))
        self.assertRaises(AssertionError, re_sample.get_text, 'y')

    def test_get_value(self):
        # test get value
        self.assertEqual(data['x'],
                         re_sample.get_value('x'))

    def test_get_dp(self):
        # test get sent ids

        deprel, head = re_sample.get_dp()

        self.assertEqual(len(data['x']),len(deprel))
        self.assertEqual(len(data['x']),len(head))

    def test_get_en(self):
        # test get en

        sh, st, oh, ot = re_sample.get_en()
        self.assertEqual(8, sh)
        self.assertEqual(8, st)
        self.assertEqual(10, oh)
        self.assertEqual(10, ot)

    def test_get_type(self):
        # test get type

        subj_type, obj_type, ner = re_sample.get_type()
        self.assertTrue(len(ner) == len(data['x']))
        self.assertTrue(isinstance(subj_type, str))
        self.assertTrue(isinstance(obj_type, str))
        self.assertTrue(isinstance(ner, list))

    def test_get_sent(self):
        # test get sent
        x, y = re_sample.get_sent()
        self.assertEqual(data['x'],
                         x)
        self.assertEqual('age', y)

    def stan_ner_transform(self):
        # test stan ner transform
        ners = re_sample.stan_ner_transform()
        self.assertTrue(len(ners) == len(re_sample.get_text('x')))

    def test_delete_field_at_indices(self):
        # test delete field at indices
        delete = re_sample.delete_field_at_indices('x', [[1, 4], 5])
        self.assertEqual(["``", "very", ",", "''", "Mattis", ",", "30", ",",
                          "told", "reporters", "after", "meeting", "with",
                          "Ban", "in", "New", "York", "."],
                         delete.dump()['x'])

    def test_insert_field_after_indices(self):
        # test insert field after indices
        ins_after = re_sample.insert_field_after_indices(
            'x', [1, 4], [['cls', 'new'], 'end'])
        self.assertEqual(["``", "The", "cls", "new", "situation", "is", "very",
                          "end", "serious", ",", "''", "Mattis", ",",  "30",
                          ",", "told", "reporters", "after", "meeting", "with",
                          "Ban", "in", "New", "York", "."],
                         ins_after.dump()['x'])

    def test_insert_field_before_indices(self):
        # test insert field before indices
        ins_before = re_sample.insert_field_before_indices(
            'x', [1, 4], [['cls', 'new'], 'end'])
        self.assertEqual(["``", "cls", "new", "The", "situation", "is", "end",
                          "very", "serious", ",", "''", "Mattis", ",", "30",
                          ",", "told", "reporters", "after", "meeting", "with",
                          "Ban", "in", "New", "York", "."],
                         ins_before.dump()['x'])

    def test_dump(self):
        # test dump
        self.assertEqual(data, re_sample.dump())


if __name__ == "__main__":
    unittest.main()

import unittest

from TextFlint.input_layer.component.sample.re_sample import RESample
from TextFlint.generation_layer.transformation.RE.insert_clause \
    import InsertClause

data = {'x': ['But', 'both', 'Renault', 'and', 'Peugeot', 'said', 'July', '19',
              ',', '2006', ',', 'Wednesday','it', 'was', 'not', 'their',
              'responsibility', 'to', 'pay', 'out', 'compensation', 'to',
              'the','Sacramento', 'County', 'Superior', 'Court',
              'workers', '.'],
            'subj': [23, 26], 'obj': [6, 11], 'y': 'no_relation'}
sample = InsertClause()
re_data = RESample(data)


class TestEntitySwap(unittest.TestCase):
    def test_get_clause(self):
        self.assertRaises(AssertionError, sample.get_clause, [])
        clauseadd = sample.get_clause('Fudan%20University')
        self.assertTrue(isinstance(clauseadd, str))

    def test_get_clause_(self):
        self.assertRaises(AssertionError, sample._get_clause, re_data)
        clauseadd = sample._get_clause('Q148')
        self.assertTrue(isinstance(clauseadd, str))

    def test_transform(self):
        self.assertRaises(AssertionError, sample._transform, [], 1)
        self.assertRaises(AssertionError, sample._transform, re_data, [])
        trans_samples = sample._transform(re_data, 1)
        for trans_sample in trans_samples:
            self.assertTrue(type(trans_sample) == type(re_data))


if __name__ == "__main__":
    unittest.main()

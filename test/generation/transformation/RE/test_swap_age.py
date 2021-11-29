import unittest

from textflint.input.component.sample.re_sample import RESample
from textflint.generation.transformation.RE.swap_age import SwapAge

data = {'x': ['Allen', 'waits', ',', '46', ',', 'was', 'also', 'convicted',
              'of', 'illegal', 'drug', 'activity', ',', 'and', 'bribing',
              'police',  'to', 'turn', 'a', 'blind', 'eye', 'to', 'her',
              'crimes', ',', 'earlier', 'press', 'reports', 'said', '.'],
        'subj': [0, 1], 'obj': [3, 3], 'y': 'age'}
sample = SwapAge()
re_data = RESample(data)


class TestEntitySwap(unittest.TestCase):
    def test_transform(self):
        self.assertRaises(AssertionError, sample._transform, [], 1)
        self.assertRaises(AssertionError, sample._transform, re_data, [])
        trans_samples = sample._transform(re_data, 1)
        for trans_sample in trans_samples:
            self.assertTrue(type(trans_sample) == type(re_data))


if __name__ == "__main__":
    unittest.main()

import random
import unittest

from textflint.input.component.sample import UTCnSample
from textflint.generation.transformation.UTCN import CnPunctuation

sample = UTCnSample({
    'x': '今天天气很好。',
    'y': 1,
})
trans_method = CnPunctuation()


class TestPunctuation(unittest.TestCase):
    def test_transformation(self):
        change_sample = trans_method.transform(sample, n=3)
        self.assertEqual(3, len(change_sample))

        for s in change_sample:
            self.assertTrue(len(s.get_value('x')) > len(sample.get_value('x')))
            self.assertTrue(s.get_value('x') != sample.get_value('x'))

if __name__ == "__main__":
    unittest.main()

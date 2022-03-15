import unittest

from textflint.input.component.sample.smcn_sample import SMCNSample
from textflint.generation.transformation.SMCN import Overlap

sample1 = SMCNSample({'sentence1': '天气预报说上海今天下雨的概率是80%。',
        'sentence2': '根据天气预报，上海今天有80%的概率下雨。',
        'y': '1'})

sm_overlap = Overlap()


class TestSmOverlap(unittest.TestCase):
    def test_label(self):
        # SmOverlap only has 2 labels: ['1', '0']
        trans = sm_overlap.transform(sample1)
        self.assertTrue(trans[0].y.field_value in ['0', '1'])


if __name__ == "__main__":
    unittest.main()

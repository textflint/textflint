import unittest

from textflint.input_layer.component.sample.sm_sample import SMSample
from textflint.generation_layer.transformation.SM import Overlap

sample1 = SMSample({'sentence1': 'MR zhang has 10 students',
        'sentence2': 'Mr zhang has 10 students',
        'y': '1'})

sm_overlap = Overlap()


class TestSmOverlap(unittest.TestCase):
    def test_label(self):
        # SmOverlap only has 2 labels: ['1', '0']
        trans = sm_overlap.transform(sample1)
        self.assertTrue(trans[0].y.field_value in ['0', '1'])


if __name__ == "__main__":
    unittest.main()

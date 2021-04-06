import unittest

from textflint.input_layer.component.sample.nli_sample import NLISample
from textflint.generation_layer.transformation.NLI import Overlap

sample1 = NLISample({'hypothesis': 'MR zhang has 10 students',
        'premise': 'Mr zhang has 10 students',
        'y': 'entailment'})

nli_overlap = Overlap()


class TestNliOverlap(unittest.TestCase):
    def test_label(self):
        # NliNumWord only has 2 labels: ['entailment', 'non-entailment']
        trans = nli_overlap.transform(sample1)
        self.assertTrue(trans[0].y.field_value in ['entailment',
                                                   'non-entailment'])


if __name__ == "__main__":
    unittest.main()
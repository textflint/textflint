import unittest

from textflint.input_layer.component.sample.nli_sample import NLISample
from textflint.generation_layer.transformation.NLI import SwapAnt

sample1 = NLISample({'hypothesis': 'There are two little boys smiling.',
        'premise': 'Two little boys are smiling and laughing while one is '
                   'standing and one is in a bouncy seat',
        'y': 'entailment'})

nli_antonymswap = SwapAnt()


class TestNliSwapAnt(unittest.TestCase):
    def test_whether_changed(self):
        # test whether the sample changed or not
        trans = nli_antonymswap.transform(sample1)
        self.assertTrue(sample1.hypothesis.field_value !=
                        trans[0].hypothesis.field_value or
                        sample1.premise.field_value !=
                        trans[0].premise.field_value)

    def test_label(self):
        # NLIAntonymSwap will change some word to its opposite meaning,
        # which leads contradiction label
        trans = nli_antonymswap.transform(sample1)
        self.assertEqual('contradiction', trans[0].y.field_value)

    def test_empty_sample(self):
        # test sample with empty string and empty list
        self.assertRaises(AttributeError, nli_antonymswap.transform, '')
        self.assertRaises(AttributeError, nli_antonymswap.transform, [])


if __name__ == "__main__":
    unittest.main()
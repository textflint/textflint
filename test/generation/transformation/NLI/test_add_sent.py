import unittest

from textflint.input.component.sample.nli_sample import NLISample
from textflint.generation.transformation.NLI import AddSent

sample1 = NLISample({
    'hypothesis': 'MR zhang has 10 students',
    'premise': 'Mr zhang has 20 students',
    'y': 'contradiction'
})

nli_length = AddSent()


class TestNliLength(unittest.TestCase):
    def test_whether_changed(self):
        # test whether the sample changed or not
        trans = nli_length.transform(sample1)
        self.assertTrue(sample1.hypothesis.field_value !=
                        trans[0].hypothesis.field_value or
                        sample1.premise.field_value !=
                        trans[0].premise.field_value)

    def test_label(self):
        # NliLength only add a sentence to hypothesis, which will
        # not change the label
        trans = nli_length.transform(sample1)
        self.assertEqual(sample1.y.field_value, trans[0].y.field_value)

    def test_length(self):
        # NliLength only add a sentence to hypothesis, which means transformed
        # sample has longer hypothesis and the length of premise stay the same
        trans = nli_length.transform(sample1)
        self.assertEqual(len(sample1.premise.field_value), len(trans[0].premise.field_value))
        self.assertTrue(len(sample1.hypothesis.words) < len(trans[0].hypothesis.words))

    def test_empty_sample(self):
        # test sample with empty string and empty list
        self.assertRaises(AttributeError, nli_length.transform, '')
        self.assertRaises(AttributeError, nli_length.transform, [])


if __name__ == "__main__":
    unittest.main()

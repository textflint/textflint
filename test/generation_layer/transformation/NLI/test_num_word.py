import unittest

from textflint.input_layer.component.sample.nli_sample import NLISample
from textflint.generation_layer.transformation.NLI import NumWord

sample1 = NLISample({'hypothesis': 'MR zhang has 10 students',
        'premise': 'Mr zhang has 10 students',
        'y': 'entailment'})

nli_numword = NumWord()


class TestNliNumWord(unittest.TestCase):
    def test_whether_changed(self):
        # test whether the sample changed or not
        trans = nli_numword.transform(sample1)
        self.assertTrue(sample1.hypothesis.field_value !=
                        trans[0].hypothesis.field_value or
                        sample1.premise.field_value !=
                        trans[0].premise.field_value)

    def test_label(self):
        # NliNumWord will change some thing about number,
        # which leads contradiction label
        trans = nli_numword.transform(sample1)
        self.assertEqual('contradiction', trans[0].y.field_value)


    def test_empty_sample(self):
        # test sample with empty string and empty list
        self.assertRaises(AttributeError, nli_numword.transform, '')
        self.assertRaises(AttributeError, nli_numword.transform, [])


if __name__ == "__main__":
    unittest.main()

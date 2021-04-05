import unittest

from TextFlint.generation_layer.subpopulation import SubPopulation
from TextFlint.input_layer.component.sample import NLISample
sent1 = 'Mr Zhang has 10 students in Fudan university.'
sent2 = 'Mr Zhang has 10 students.'
sample1 = NLISample({
    'hypothesis': sent1,
    'premise': sent2,
    'y': 'entailment'
})


class TestSubPopulation(unittest.TestCase):

    def test_normalize_bound(self):
        self.assertEqual(SubPopulation.normalize_bound(0.3, 100), 30)
        self.assertEqual(SubPopulation.normalize_bound("30%", 100), 30)


if __name__ == "__main__":
    unittest.main()

import unittest

from textflint.generation_layer.subpopulation.UT import LMSubPopulation
from textflint.input_layer.component.sample import NLISample
sent1 = 'The man was carrying a red box.'
sent2 = 'Man with a blue box on wheels smiling at the boys playing around him.'
sample1 = NLISample({
    'hypothesis': sent1,
    'premise': sent2,
    'y': 'entailment'
})

sub = LMSubPopulation(intervals=[2, 4])


class TestLengthSubPopulation(unittest.TestCase):

    def test_lm_score(self):
        score = sub._score(sample1, fields=['hypothesis', 'premise'])
        self.assertTrue(abs(score-11634) < 1)


if __name__ == "__main__":
    unittest.main()

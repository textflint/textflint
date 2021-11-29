import unittest

from textflint.generation.subpopulation.UT import PhraseSubPopulation
from textflint.input.component.sample import NLISample
sent1 = 'Mr Zhang has 10 students in Fudan university.'
sent2 = 'Mr Zhang has 10 students.'
sample1 = NLISample({
    'hypothesis': sent1,
    'premise': sent2,
    'y': 'entailment'
})

sub = PhraseSubPopulation(phrase_name='negation')


class TestPhraseSubPopulation(unittest.TestCase):

    def test_length_score(self):
        score = sub._score(sample1, fields=['hypothesis', 'premise'])
        self.assertEqual(score, 0)


if __name__ == "__main__":
    unittest.main()

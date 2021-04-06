import unittest

from textflint.generation_layer.subpopulation.UT import PrejudiceSubPopulation
from textflint.input_layer.component.sample import NLISample

sub = PrejudiceSubPopulation(mode='man')


class TestPrejudiceSubPopulation(unittest.TestCase):

    def test_prejudice_score(self):
        score1 = sub._score(
            NLISample({'hypothesis': "A young boy",
                       'premise': "A young girl", 'y': 'entailment'}),
                            fields=['hypothesis', 'premise'])
        score2 = sub._score(NLISample(
            {'hypothesis': "A young boy",
             'premise': "A young boy", 'y': 'entailment'}),
                            fields=['hypothesis', 'premise'])
        self.assertEqual(score1, 0)
        self.assertEqual(score2, 1)


if __name__ == "__main__":
    unittest.main()

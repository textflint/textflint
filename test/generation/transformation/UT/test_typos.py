import unittest
import random
import editdistance

from textflint.input.component.sample import SASample
from textflint.generation.transformation.UT.typos import Typos


sample = {'x': 'Pride and Prejudice is a famous fiction', 'y': 'positive'}
data_sample = SASample(sample)
typos_trans = Typos(mode='random')
random.seed(100)


class TestTwitterType(unittest.TestCase):
    def test_random(self):
        typos_trans.mode = 'random'
        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], typos_trans.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual(1, len(typos_trans.transform(special_sample)))

        x = typos_trans.transform(data_sample, n=3)
        self.assertTrue(3 == len(x))

        change = []
        for sample in x:
            self.assertTrue(sample.get_text('x') != data_sample.get_text('x'))
            self.assertTrue(editdistance.distance(sample.get_text('x'),
                                                data_sample.get_text('x')) <= 4)
            change.append(sample.get_text('x'))

        self.assertTrue(len(set(change)) == 3)

    def test_rep(self):
        typos_trans.mode = 'replace'
        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], typos_trans.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual(1, len(typos_trans.transform(special_sample)))

        x = typos_trans.transform(data_sample, n=3)
        self.assertTrue(3 == len(x))

        change = []
        for sample in x:
            self.assertTrue(sample.get_text('x') != data_sample.get_text('x'))
            self.assertTrue(editdistance.distance(sample.get_text('x'),
                                                data_sample.get_text('x')) <= 2)
            change.append(sample.get_text('x'))

        self.assertTrue(len(set(change)) == 3)

    def test_insert(self):
        typos_trans.mode = 'insert'
        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], typos_trans.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual(1, len(typos_trans.transform(special_sample)))

        x = typos_trans.transform(data_sample, n=3)
        self.assertTrue(3 == len(x))

        change = []
        for sample in x:
            self.assertTrue(sample.get_text('x') != data_sample.get_text('x'))
            self.assertTrue(editdistance.distance(sample.get_text('x'),
                                                data_sample.get_text('x')) <= 2)
            change.append(sample.get_text('x'))

        self.assertTrue(len(set(change)) == 3)

    def test_delete(self):
        typos_trans.mode = 'delete'

        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], typos_trans.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual(1, len(typos_trans.transform(special_sample)))

        x = typos_trans.transform(data_sample, n=3)
        self.assertTrue(3 == len(x))

        change = []
        for sample in x:
            self.assertTrue(sample.get_text('x') != data_sample.get_text('x'))
            self.assertTrue(editdistance.distance(sample.get_text('x'),
                                                data_sample.get_text('x')) <= 2)
            change.append(sample.get_text('x'))

        self.assertTrue(len(set(change)) == 3)


    def test_swap(self):
        typos_trans.mode = 'swap'
        x = typos_trans.transform(data_sample, n=1)
        self.assertTrue(1 == len(x))

        for sample in x:
            self.assertTrue(sample.get_text('x') != data_sample.get_text('x'))
            self.assertTrue(editdistance.distance(sample.get_text('x'),
                                                data_sample.get_text('x')) <= 4)

        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], typos_trans.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual(1, len(typos_trans.transform(special_sample)))


if __name__ == "__main__":
    unittest.main()

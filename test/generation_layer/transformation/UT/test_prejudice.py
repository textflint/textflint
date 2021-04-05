import unittest

from TextFlint.input_layer.component.sample import SASample
from TextFlint.generation_layer.transformation.UT.prejudice import Prejudice


class TestPrejudice(unittest.TestCase):
    def test_transformation(self):
        # test wrong mode
        self.assertRaises(ValueError, Prejudice, 'Loc', 'woman')
        self.assertRaises(ValueError, Prejudice, 'Name', 'Japan')
        self.assertRaises(ValueError, Prejudice, 'Loc', 'Ja')
        self.assertRaises(ValueError, Prejudice, 'Loc', ['Ja'])

        import random
        random.seed(100)

        sent1 = "Interesting and moving performances by Tom Courtenay " \
                "and Peter Finch"
        swap_ins = Prejudice(
            change_type='Name',
            prejudice_tendency='woman')

        data_sample = SASample({'x': sent1, 'y': "negative"})
        x = swap_ins.transform(data_sample, n=5)

        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))

        change = []
        for sa_sample in x:
            self.assertEqual(data_sample.get_words('x')[:5],
                             sa_sample.get_words('x')[:5])
            self.assertEqual(data_sample.get_words('x')[6:8],
                             sa_sample.get_words('x')[6:8])
            self.assertEqual(data_sample.get_words('x')[-1],
                             sa_sample.get_words('x')[-1])
            change.append(sa_sample.get_words('x')[5] +
                          sa_sample.get_words('x')[8])

        self.assertTrue(5 == len(set(change)))


if __name__ == "__main__":
    unittest.main()

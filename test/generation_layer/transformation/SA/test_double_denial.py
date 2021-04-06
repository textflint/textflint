import unittest

from textflint.input_layer.component.sample.sa_sample import SASample
from textflint.generation_layer.transformation.SA.double_denial \
    import DoubleDenial

data = {'x': "It's a Bad movie. Bad actor",
        'y': 'negative'}
sa_sample = SASample(data)
swap_ins = DoubleDenial()


class TestDoubleDenial(unittest.TestCase):
    def test_get_double_denial_info(self):
        # test data with two special words
        tokens = ["bad", "guy", "good", "guy"]
        self.assertEqual(len(swap_ins._get_double_denial_info(tokens)[0]),
                         len(swap_ins._get_double_denial_info(tokens)[1]))
        self.assertEqual(len(swap_ins._get_double_denial_info(tokens)[0]), 2)

        # test data with no special words
        tokens = [1, 2, 3]
        self.assertEqual(len(swap_ins._get_double_denial_info(tokens)[0]),
                         len(swap_ins._get_double_denial_info(tokens)[1]))
        self.assertEqual(len(swap_ins._get_double_denial_info(tokens)[0]), 0)

    def test_DoubleDenial(self):
        # test data with two special words
        self.assertEqual(len(swap_ins.transform(sa_sample)), 1)

        # test data with no special words
        test_data = {'x': "It's a movie.  actor",
                     'y': 'negative'}
        test_sample = SASample(test_data)
        self.assertEqual([], swap_ins.transform(test_sample))

        test_data = {'x': "It's a go od movie.  ba d actor",
                     'y': 'negative'}
        test_sample = SASample(test_data)
        self.assertEqual([], swap_ins.transform(test_sample))


if __name__ == "__main__":
    unittest.main()

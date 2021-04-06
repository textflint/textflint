import unittest

from textflint.input_layer.component.sample import SASample
from textflint.generation_layer.transformation.UT.contraction import Contraction

sent1 = "we're playing ping pang ball, you are so lazy. She's so beautiful!"
data_sample = SASample({'x': sent1, 'y': "negative"})
swap_ins = Contraction()


class TestContraction(unittest.TestCase):
    def test_get_contractions(self):
        self.assertTrue(([[1, 3]], ['cannot']) ==
                        swap_ins._get_contractions(['I', 'can', "'t", 'do']))
        self.assertEqual(([[0, 2]], ['I am']),
                         swap_ins._get_contractions(['I', "'m", "'a", 'student']))
        self.assertEqual(([], []), swap_ins._get_contractions([]))

    def test_get_expanded_phrases(self):
        self.assertEqual(
            ([], []), swap_ins._get_expanded_phrases(['I', 'can', "'t", 'do']))

    def test_transformation(self):
        trans = swap_ins.transform(data_sample)
        self.assertEqual(1, len(trans))
        trans = trans[0]
        self.assertEqual(
            "we are playing ping pang ball, you're so lazy. "
            "She is so beautiful!", trans.get_text('x'))
        self.assertEqual(trans.get_value('y'), data_sample.get_value('y'))

        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))


if __name__ == "__main__":
    unittest.main()

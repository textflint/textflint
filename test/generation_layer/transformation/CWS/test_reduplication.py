import unittest

from textflint.input_layer.component.sample.cws_sample import CWSSample
from textflint.generation_layer.transformation.CWS.reduplication \
    import Reduplication

sent1 = '朦胧的月色'
sample = CWSSample({'x': sent1, 'y': ['B', 'E', 'S', 'B', 'E']})
swap_ins = Reduplication()


class TestReduplication(unittest.TestCase):
    def test_transformation(self):
        trans_sample = swap_ins.transform(sample)
        self.assertTrue(1 == len(trans_sample))
        trans_sample = trans_sample[0]
        self.assertEqual('朦朦胧胧的月色', trans_sample.get_value('x'))
        self.assertEqual(['B', 'M', 'M', 'E', 'S', 'B', 'E'],
                         trans_sample.get_value('y'))
        self.assertEqual([2, 2, 2, 2, 0, 0, 0], trans_sample.mask)


if __name__ == "__main__":
    unittest.main()

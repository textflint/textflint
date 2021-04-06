import unittest

from textflint.input_layer.component.sample import SASample
from textflint.generation_layer.transformation.UT.insert_adv import InsertAdv

sent1 = 'The quick brown fox jumps over the lazy dog .'
data_sample = SASample({'x': sent1, 'y': "negative"})
swap_ins = InsertAdv()


class TestAddAdverb(unittest.TestCase):
    def test_transformation(self):
        # test the change num
        change_sample = swap_ins.transform(data_sample, n=10)
        self.assertEqual(10, len(change_sample))

        special_sample = SASample(
            {'x': '', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))

        # test if the item change
        change_item = []
        for sample in change_sample:
            self.assertEqual(data_sample.get_words('x')[:4],
                             sample.get_words('x')[:4])
            self.assertEqual(data_sample.get_words('x')[-6:],
                             sample.get_words('x')[-6:])
            self.assertEqual(len(data_sample.get_words('x')) + 1,
                             len(sample.get_words('x')))
            change_item.append(sample.get_words('x')[4])

        # test if the n change samples not equal
        self.assertEqual(len(set(change_item)), 10)


if __name__ == "__main__":
    unittest.main()

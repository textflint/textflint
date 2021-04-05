import unittest

from TextFlint.generation_layer.transformation.CWS.swap_name import SwapName
from TextFlint.input_layer.component.sample.cws_sample import CWSSample

sent1 = ['我朝', '小明走了过去']
data_sample = CWSSample({'x': sent1, 'y':
    ['S', 'S', 'B', 'E', 'S', 'S', 'B', 'E']})
swap_ins = SwapName()


class TestSwapName(unittest.TestCase):
    def test_transformation(self):
        trans_sample = swap_ins.transform(data_sample, n=10)
        self.assertTrue(10 == len(trans_sample))
        for sample in trans_sample:
            for i, (trans_word, ori_word) in \
                    enumerate(zip(sample.get_value('x'),
                                  data_sample.get_value('x'))):
                self.assertTrue(sample.get_value('y') ==
                                data_sample.get_value('y'))
                if not sample.mask[i]:
                    self.assertTrue(trans_word == ori_word)


if __name__ == "__main__":
    unittest.main()

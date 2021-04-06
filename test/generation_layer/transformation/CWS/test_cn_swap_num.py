import unittest

from textflint.generation_layer.transformation.CWS.cn_swap_num import CnSwapNum
from textflint.input_layer.component.sample.cws_sample import CWSSample

sent1 = '小明 买书 花 了 九百'
data_sample = CWSSample({'x': sent1, 'y': []})
swap_ins = CnSwapNum()


class TestCnSwapNum(unittest.TestCase):
    def test_crete_number(self):
        for i in range(10):
            self.assertTrue(len(swap_ins.create_num(i)) == 1)
        self.assertTrue(3 == len(swap_ins.create_num(10)))
        self.assertTrue(5 == len(swap_ins.create_num(11)))
        self.assertTrue(7 == len(swap_ins.create_num(12)))

    def test_compare(self):
        self.assertTrue(swap_ins.compare('九百三十一', '八百三十二'))
        self.assertTrue(not swap_ins.compare('三十一', '三十二'))
        self.assertTrue(not swap_ins.compare('十一', '十二'))
        self.assertTrue(not swap_ins.compare('一', '二'))

    def test_number_change(self):
        for i in range(10):
            change, change_label = swap_ins.number_change('一万',
                                                          ['B', 'E'], 0, 1)
            self.assertTrue(len(change) == len(change_label))

    def test_transformation(self):
        trans_sample = swap_ins.transform(data_sample, n=10)
        for sample in trans_sample:
            for i in range(len(data_sample.get_value('x'))):
                if sample.mask[i] == 0:
                    self.assertTrue(sample.get_value('x')[i] ==
                                    data_sample.get_value('x')[i])
            sample.dump()


if __name__ == "__main__":
    unittest.main()

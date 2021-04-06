import unittest

from textflint.generation_layer.transformation.CWS.cn_mlm import CnMLM
from textflint.input_layer.component.sample.cws_sample import CWSSample

sent1 = '小明 喜欢 看书 。'
data_sample = CWSSample({'x': sent1, 'y': []})
swap_ins = CnMLM()


class TestAbbreviation(unittest.TestCase):
    def test_check(self):
        self.assertTrue(swap_ins.check(0, 1, [0, 0, 1]))
        self.assertTrue(not swap_ins.check(0, 1, [0, 1, 1]))
        self.assertTrue(not swap_ins.check(0, 1, [1, 0, 1]))

    def test_is_word(self):
        self.assertTrue(not swap_ins.is_word('飞机'))
        self.assertTrue(swap_ins.is_word('看书'))
        self.assertTrue(swap_ins.is_word('人人'))
        self.assertTrue(not swap_ins.is_word('汽车'))

    def test_create_word(self):
        self.assertTrue(swap_ins.create_word('小 明 在 看 [MASK] [MASK] 。'))
        self.assertTrue(not swap_ins.create_word('小 明 在 看 [MASK] [MASK]'))

    def test_transform(self):
        trans_data = swap_ins.transform(data_sample)
        self.assertTrue(len(trans_data) == 1)
        self.assertEqual('小明喜欢看电影。', trans_data[0].get_value('x'))
        self.assertEqual(['B', 'E', 'B', 'E', 'S', 'B', 'E', 'S'],
                         trans_data[0].get_value('y'))
        self.assertTrue([0, 0, 0, 0, 0, 2, 2, 0] == trans_data[0].mask)


if __name__ == "__main__":
    unittest.main()

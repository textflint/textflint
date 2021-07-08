import unittest

from textflint.generation.transformation.CWS.cn_mlm import CnMLM
from textflint.input.component.sample.cws_sample import CWSSample

sent1 = '小明 喜欢 看书 。'
data_sample = CWSSample({'x': sent1, 'y': []})
swap_ins = CnMLM()


class TestCnMLM(unittest.TestCase):
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
        trans_data = swap_ins.transform(
            CWSSample({'x': '玩具厂 大量 生产 玩具 。', 'y': []}))
        self.assertEqual(1, len(trans_data))


if __name__ == "__main__":
    unittest.main()

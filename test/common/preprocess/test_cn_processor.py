import unittest

from TextFlint.common.preprocess.cn_processor import CnProcessor


class TestCnProcessor(unittest.TestCase):
    test_processor = CnProcessor()

    def test_cn_processor(self):
        sent = '小明想去吃螺蛳粉'
        self.assertRaises(AssertionError, self.test_processor.get_ner, {})
        self.assertRaises(AssertionError, self.test_processor.get_pos_tag, {})
        self.assertEqual(([('Nh', 0, 1)],
                          ['Nh', 'Nh', 'O', 'O', 'O', 'O', 'O', 'O']),
                         self.test_processor.get_ner(sent))
        self.assertEqual([['nh', 0, 1], ['v', 2, 2],
                          ['v', 3, 3], ['v', 4, 4], ['n', 5, 7]],
                         self.test_processor.get_pos_tag(sent))


if __name__ == "__main__":
    unittest.main()

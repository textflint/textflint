import unittest

from textflint.input_layer.component.sample.cws_sample import *


class TestField(unittest.TestCase):
    def test_field(self):
        # test wrong data
        self.assertRaises(AssertionError, CWSSample, {'x': '小明'})
        self.assertRaises(AssertionError, CWSSample, {'x': {}, 'y': []})
        self.assertRaises(AssertionError, CWSSample, {'x': '小明', 'y': ['B']})
        self.assertRaises(AssertionError, CWSSample, {'x': '小明',
                                                      'y': ['B', 'Y']})

        test_sample = CWSSample({'x': '小明 好想 送 Jo 圣诞 礼物', 'y': []})

        # test pos tag only the return format is correct, not the label
        pos_tag = test_sample.pos_tags
        self.assertEqual(pos_tag[-1][-1] + 1, len(test_sample.get_value('x')))
        self.assertEqual(pos_tag[0][1], 0)
        for tag in pos_tag:
            self.assertTrue([str, int, int], [type(i) for i in tag])

        # test ner only the return format is correct, not the label
        ner, ner_label = test_sample.ner
        self.assertEqual(len(ner_label), len(test_sample.get_value('x')))
        for tag in ner:
            self.assertTrue([str, int, int] == [type(k) for k in tag]
                            and tag[1] <= tag[2])

        # test get_word
        self.assertEqual(['小明', '好想', '送', 'Jo', '圣诞', '礼物'],
                         test_sample.get_words())

        # test replace_at_ranges
        new = test_sample.replace_at_ranges([[0, 2]], ['李明'], [['B', 'E']])
        self.assertEqual('李明好想送Jo圣诞礼物', new.get_value('x'))
        # test wrong answer
        self.assertRaises(ValueError, test_sample.replace_at_ranges, [[2, 0]],
                          ['李明'], [['B', 'E']])
        self.assertRaises(AssertionError, test_sample.replace_at_ranges,
                          [[0, 2], []], ['李明'], [['B', 'E']])

        # check label
        self.assertEqual(['B', 'E', 'B', 'E', 'S', 'B', 'E', 'B',
                          'E', 'B', 'E'], new.get_value('y'))

        # check mask
        self.assertEqual([2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], new.mask)

        # get_labels
        self.assertEqual(['S'], new.get_labels('一'))
        self.assertEqual(['B', 'M', 'E'], new.get_labels('一二三'))


if __name__ == "__main__":
    unittest.main()

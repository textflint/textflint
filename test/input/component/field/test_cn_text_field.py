import unittest

from textflint.input.component.field.cn_text_field import *

sents = [
    "我们正在玩乒乓球，你真的太懒了。她打的很漂亮。",
    '我不理解这个问题！',
    "2021年计划我们写了三张纸。"
]


class TestCnTextField(unittest.TestCase):
    def test_cn_text_field(self):
        self.assertRaises(ValueError, CnTextField, {})

        # test mask
        test_field = CnTextField('小明去上海')

        # test pos tag only the return format is correct, not the label
        pos_tag = test_field.pos_tags()
        self.assertEqual(pos_tag[-1][-1] + 1, len(test_field.tokens))
        self.assertEqual(pos_tag[0][1], 0)
        for tag in pos_tag:
            self.assertTrue([str, int, int], [type(i) for i in tag])

        # test ner only the return format is correct, not the label
        ner, ner_label = test_field.ner()
        self.assertEqual(len(ner_label), len(test_field.tokens))
        for tag in ner:
            self.assertTrue([str, int, int] == [type(k) for k in tag] and tag[1] <= tag[2])

    def test_words_length(self):
        test_field = CnTextField('小明去上海')
        self.assertRaises(ValueError, CnTextField, {})
        self.assertEqual(3, test_field.words_length)

        test_field = CnTextField(sents[0])
        self.assertEqual(17, test_field.words_length)

        test_field = CnTextField('')
        self.assertEqual(0, test_field.words_length)

    def test_text(self):
        test_field = CnTextField(sents[0])
        self.assertEqual(sents[0], test_field.text)

    def test_mask(self):
        test_field = CnTextField(sents[0])
        self.assertEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], test_field.mask)

    def test_tokens(self):
        for sent in sents:
            test_field = CnTextField(sent)
            self.assertEqual([k for k in sent], test_field.tokens)

    def test_ner(self):
        sent = '今天天气很好。'
        test_field = CnTextField(sent)
        self.assertEqual(([], ['O', 'O', 'O', 'O', 'O', 'O', 'O']), test_field.ner())
        sent = ''
        test_field = CnTextField(sent)
        self.assertEqual(([], []), test_field.ner())

    def test_pos_tags(self):
        sent = '今天天气很好。'
        test_field = CnTextField(sent)
        self.assertEqual([['nt', 0, 1], ['n', 2, 3], ['d', 4, 4], ['a', 5, 5], ['wp', 6, 6]], test_field.pos_tags())
        sent = ''
        test_field = CnTextField(sent)
        self.assertEqual([], test_field.pos_tags())

    def test_dp(self):
        sent = '今天天气很好。'
        test_field = CnTextField(sent)
        self.assertEqual([(1, 2, 'ATT'), (2, 4, 'SBV'), (3, 4, 'ADV'), (4, 0, 'HED'), (5, 4, 'WP')], test_field.dp())
        sent = ''
        test_field = CnTextField(sent)
        self.assertEqual([], test_field.dp())

    def test_insert_after_index(self):
        test_field = CnTextField(sents[0])
        self.assertRaises(AssertionError, test_field.insert_after_index, 5, '')

        self.assertEqual('我们正在好的玩乒乓球，你真的太懒了。她打的很漂亮。', test_field.insert_after_index(3, '好的').text)
        self.assertEqual('我们正在好的玩乒乓球，你真的太懒了。她打的很漂亮。', test_field.insert_after_index(3, ['好的']).text)
        self.assertEqual('我们正在好的玩乒乓球，你真的太懒了。她打的很漂亮。', test_field.insert_after_index(3, ['好','的']).text)

    def test_insert_after_indices(self):
        test_field = CnTextField(sents[0])
        self.assertRaises(AssertionError, test_field.insert_after_indices, 5, '')
        self.assertRaises(AssertionError, test_field.insert_after_indices, [5], [''])
        self.assertRaises(AssertionError, test_field.insert_after_indices, [5], ['1','5'])

        self.assertEqual('我们正在好的玩乒干嘛乓球，你真的太懒了。她打的很漂亮。', test_field.insert_after_indices([3,5], ['好的','干嘛']).text)
        self.assertEqual('我们正在好的玩乒干嘛乓球，你真的太懒了。她打的很漂亮。', test_field.insert_after_indices([3,5], [['好的'],'干嘛']).text)
        self.assertEqual('我们正在好的玩乒干嘛乓球，你真的太懒了。她打的很漂亮。', test_field.insert_after_indices([3,5], [['好','的'],'干嘛']).text)
    def test_insert_before_index(self):
        test_field = CnTextField(sents[0])
        self.assertRaises(AssertionError, test_field.insert_before_index, 5, '')

        self.assertEqual('我们正好的在玩乒乓球，你真的太懒了。她打的很漂亮。', test_field.insert_before_index(3, '好的').text)
        self.assertEqual('我们正好的在玩乒乓球，你真的太懒了。她打的很漂亮。', test_field.insert_before_index(3, ['好的']).text)
        self.assertEqual('我们正好的在玩乒乓球，你真的太懒了。她打的很漂亮。', test_field.insert_before_index(3, ['好','的']).text)

    def test_insert_before_indices(self):
        test_field = CnTextField(sents[0])
        self.assertRaises(AssertionError, test_field.insert_before_indices, 5, '')
        self.assertRaises(AssertionError, test_field.insert_before_indices, [5], [''])
        self.assertRaises(AssertionError, test_field.insert_before_indices, [5], ['1','5'])

        self.assertEqual('我们正好的在玩干嘛乒乓球，你真的太懒了。她打的很漂亮。', test_field.insert_before_indices([3,5], ['好的','干嘛']).text)
        self.assertEqual('我们正好的在玩干嘛乒乓球，你真的太懒了。她打的很漂亮。', test_field.insert_before_indices([3,5], [['好的'],'干嘛']).text)
        self.assertEqual('我们正好的在玩干嘛乒乓球，你真的太懒了。她打的很漂亮。', test_field.insert_before_indices([3,5], [['好','的'],'干嘛']).text)

    def test_swap_at_index(self):
        test_field = CnTextField(sents[0])
        self.assertRaises(TypeError, test_field.swap_at_index)
        self.assertRaises(ValueError, test_field.swap_at_index,1,1)
        self.assertRaises(AssertionError, test_field.swap_at_index,[5,7],[1,3])
        self.assertEqual('我们正乒玩在乓球，你真的太懒了。她打的很漂亮。',test_field.swap_at_index(3, 5).text)

    def test_delete_at_index(self):
        test_field = CnTextField(sents[0])
        self.assertRaises(ValueError, test_field.delete_at_index,200)
        self.assertEqual('我们正在玩乓球，你真的太懒了。她打的很漂亮。',test_field.delete_at_index(5).text)

    def test_delete_at_indices(self):
        test_field = CnTextField(sents[0])
        self.assertRaises(ValueError, test_field.delete_at_indices,[])
        self.assertRaises(ValueError, test_field.delete_at_indices,[500,700])
        self.assertEqual('我乒乓球，你真的太懒了。她打的很漂亮。',test_field.delete_at_index([1,5]).text)

    def test_replace_at_indices(self):
        test_field = CnTextField(sents[0])
        self.assertRaises(AssertionError, test_field.replace_at_indices,5,'好')
        self.assertRaises(AssertionError, test_field.replace_at_indices,[20,5],['好'])
        self.assertEqual('我们正在玩好的球，你真的太懒了。她打的很漂亮。',test_field.replace_at_indices([5,6],['好','的']).text)

    def test_replace_at_index(self):
        test_field = CnTextField(sents[0])
        self.assertRaises(AssertionError, test_field.replace_at_index,5,'好的')
        self.assertEqual('我们正在玩好乓球，你真的太懒了。她打的很漂亮。',test_field.replace_at_index(5,'好').text)


if __name__ == "__main__":
    unittest.main()

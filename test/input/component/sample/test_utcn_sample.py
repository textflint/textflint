import unittest

from textflint.input.component.sample.utcn_sample import *

sample = UTCnSample({'x': '今天天气很好。', 'y': 1})

class TestUTCnSample(unittest.TestCase):
    def test_get_value(self):
        self.assertEqual('今天天气很好。', sample.get_value('x'))
        new_sample = UTCnSample({'x': ''})
        self.assertEqual('', new_sample.get_value('x'))

    def test_get_words(self):
        self.assertEqual(['今天', '天气', '很', '好', '。'], sample.get_words('x'))
        new_sample = UTCnSample({'x': ''})
        self.assertEqual([], new_sample.get_words('x'))

    def test_get_text(self):
        self.assertEqual('今天天气很好。', sample.get_text('x'))
        new_sample = UTCnSample({'x': ''})
        self.assertEqual('', new_sample.get_text('x'))

    def test_get_mask(self):
        self.assertEqual([0, 0, 0, 0, 0, 0, 0], sample.get_mask('x'))
        new_sample = UTCnSample({'x': ''})
        self.assertEqual([], new_sample.get_mask('x'))

    def test_get_sentences(self):
        self.assertEqual(['今天天气很好。'], sample.get_sentences('x'))
        new_sample = UTCnSample({'x': ''})
        self.assertEqual([], new_sample.get_sentences('x'))

    def test_get_tokens(self):
        self.assertEqual(['今', '天', '天', '气', '很', '好', '。'], sample.get_tokens('x'))
        new_sample = UTCnSample({'x': ''})
        self.assertEqual([], new_sample.get_tokens('x'))

    def test_get_pos(self):
        self.assertEqual([['nt', 0, 1], ['n', 2, 3], ['d', 4, 4], ['a', 5, 5], ['wp', 6, 6]], sample.get_pos('x'))
        new_sample = UTCnSample({'x': ''})
        self.assertEqual([], new_sample.get_pos('x'))

    def test_get_ner(self):
        self.assertEqual(([], ['O', 'O', 'O', 'O', 'O', 'O', 'O']), sample.get_ner('x'))
        new_sample = UTCnSample({'x': ''})
        self.assertEqual(([], []), new_sample.get_ner('x'))

    def test_get_dp(self):
        self.assertEqual([(1, 2, 'ATT'), (2, 4, 'SBV'), (3, 4, 'ADV'), (4, 0, 'HED'), (5, 4, 'WP')], sample.get_dp('x'))
        new_sample = UTCnSample({'x': ''})
        self.assertEqual([], new_sample.get_dp('x'))

    def test_unequal_replace_field_at_indices(self):
        self.assertRaises(AssertionError, sample.unequal_replace_field_at_indices, 'x', [1], ['明', '天'])
        self.assertRaises(TypeError, sample.unequal_replace_field_at_indices, 'x', [0, 1.5], ['明', '天'])
        self.assertRaises(ValueError, sample.unequal_replace_field_at_indices, 'x', [1, 100], ['明', '天'])


        new_sample = sample.unequal_replace_field_at_indices('x', [(0, 2)], ['明天'])
        self.assertEqual('明天天气很好。', new_sample.get_text('x'))
        self.assertEqual([2, 2, 0, 0, 0, 0, 0], new_sample.get_mask('x'))

        new_sample = sample.unequal_replace_field_at_indices('x', [5], ['棒棒'])
        self.assertEqual('今天天气很棒棒。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 0, 0, 0, 2, 2, 0], new_sample.get_mask('x'))

        new_sample = sample.unequal_replace_field_at_indices('x', [(0, 1), (4, 6)], ['明', '棒棒'])
        self.assertEqual('明天天气棒棒。', new_sample.get_text('x'))
        self.assertEqual([2, 0, 0, 0, 2, 2, 0], new_sample.get_mask('x'))

    def test_delete_field_at_indices(self):
        self.assertRaises(AssertionError, sample.delete_field_at_indices, 'x', 100)
        self.assertRaises(ValueError, sample.delete_field_at_indices, 'x', [-1, -2])
        self.assertRaises(TypeError, sample.delete_field_at_indices, 'x', [1.5, 2])

        new_sample = sample.delete_field_at_indices('x', [1])
        self.assertEqual('今天气很好。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 0, 0, 0, 0], new_sample.get_mask('x'))

        new_sample = sample.delete_field_at_indices('x', [0, 1])
        self.assertEqual('天气很好。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 0, 0, 0], new_sample.get_mask('x'))

        new_sample = sample.delete_field_at_indices('x', [(0, 2), (4, 6)])
        self.assertEqual('天气。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 0], new_sample.get_mask('x'))

    def test_delete_field_at_index(self):
        self.assertRaises(ValueError, sample.delete_field_at_index, 'x', 100)
        self.assertRaises(ValueError, sample.delete_field_at_index, 'x', -1)
        self.assertRaises(TypeError, sample.delete_field_at_index, 'x', 1.5)

        new_sample = sample.delete_field_at_index('x', 1)
        self.assertEqual('今天气很好。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 0, 0, 0, 0], new_sample.get_mask('x'))

    def test_insert_field_before_indices(self):
        self.assertRaises(IndexError, sample.insert_field_before_indices, 'x', [100], ['确实'])
        self.assertRaises(IndexError, sample.insert_field_before_indices, 'x', [100, -1], ['确实', '确实'])
        self.assertRaises(ValueError, sample.insert_field_before_indices, 'x', [], [])
        self.assertRaises(AssertionError, sample.insert_field_before_indices, 'x', [1, 2], ['确实'])

        new_sample = sample.insert_field_before_indices('x', [2], ['确实'])
        self.assertEqual('今天确实天气很好。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 2, 2, 0, 0, 0, 0, 0], new_sample.get_mask('x'))

        new_sample = sample.insert_field_before_indices('x', [2], [['确', '实']])
        self.assertEqual('今天确实天气很好。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 2, 2, 0, 0, 0, 0, 0], new_sample.get_mask('x'))

        new_sample = sample.insert_field_before_indices('x', [2, 4], ['确实', '很'])
        self.assertEqual('今天确实天气很很好。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 2, 2, 0, 0, 2, 0, 0, 0], new_sample.get_mask('x'))

    def test_insert_field_before_index(self):
        self.assertRaises(IndexError, sample.insert_field_before_index, 'x', 100, '确实')
        self.assertRaises(ValueError, sample.insert_field_before_index, 'x', -1, '确实')
        self.assertRaises(TypeError, sample.insert_field_before_index, 'x', 1.5, '确实')

        new_sample = sample.insert_field_before_index('x', 2, '确实')
        self.assertEqual('今天确实天气很好。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 2, 2, 0, 0, 0, 0, 0], new_sample.get_mask('x'))

        new_sample = sample.insert_field_before_index('x', 2, ['确', '实'])
        self.assertEqual('今天确实天气很好。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 2, 2, 0, 0, 0, 0, 0], new_sample.get_mask('x'))

    def test_insert_field_after_indices(self):
        self.assertRaises(IndexError, sample.insert_field_after_indices, 'x', [100], ['确实'])
        self.assertRaises(IndexError, sample.insert_field_after_indices, 'x', [100, -1], ['确实', '确实'])
        self.assertRaises(ValueError, sample.insert_field_after_indices, 'x', [], [])
        self.assertRaises(AssertionError, sample.insert_field_after_indices, 'x', [1, 2], ['确实'])
        # 今天天气很好。
        new_sample = sample.insert_field_after_indices('x', [1], ['确实'])
        self.assertEqual('今天确实天气很好。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 2, 2, 0, 0, 0, 0, 0], new_sample.get_mask('x'))

        new_sample = sample.insert_field_after_indices('x', [1], [['确', '实']])
        self.assertEqual('今天确实天气很好。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 2, 2, 0, 0, 0, 0, 0], new_sample.get_mask('x'))

        new_sample = sample.insert_field_after_indices('x', [1, 3], ['确实', '很'])
        self.assertEqual('今天确实天气很很好。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 2, 2, 0, 0, 2, 0, 0, 0], new_sample.get_mask('x'))

    def test_insert_field_after_index(self):
        self.assertRaises(IndexError, sample.insert_field_after_index, 'x', 100, '确实')
        self.assertRaises(ValueError, sample.insert_field_after_index, 'x', -1, '确实')
        self.assertRaises(TypeError, sample.insert_field_after_index, 'x', 1.5, '确实')

        # new_sample = sample.insert_field_after_index('x', 1, '')
        # self.assertEqual('今天天气很好。', new_sample.get_text('x'))
        # self.assertEqual([0, 0, 0, 0, 0, 0, 0], new_sample.get_mask('x'))

        new_sample = sample.insert_field_after_index('x', 1, '确实')
        self.assertEqual('今天确实天气很好。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 2, 2, 0, 0, 0, 0, 0], new_sample.get_mask('x'))

        new_sample = sample.insert_field_after_index('x', 1, ['确', '实'])
        self.assertEqual('今天确实天气很好。', new_sample.get_text('x'))
        self.assertEqual([0, 0, 2, 2, 0, 0, 0, 0, 0], new_sample.get_mask('x'))

    def test_swap_field_at_index(self):
        self.assertRaises(TypeError, sample.swap_field_at_index, 'x', 1, '我')
        self.assertRaises(ValueError, sample.swap_field_at_index, 'x', 1, 100)

        new_sample = sample.swap_field_at_index('x', 0, 5)
        self.assertEqual('好天天气很今。', new_sample.get_text('x'))
        self.assertEqual([2, 0, 0, 0, 0, 2, 0], new_sample.get_mask('x'))

if __name__ == "__main__":
    unittest.main()

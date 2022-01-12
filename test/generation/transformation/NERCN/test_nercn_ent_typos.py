import unittest

from textflint.input.component.sample.nercn_sample import NERCnSample
from textflint.generation.transformation.NERCN.ent_typos import EntTypos,_replace_cn
from textflint.common.utils.word_op import *

sent1 = '上海浦东开发与经济建设同步'
sent_list = ['上','海','浦','东','开','发','与','法','制','建','设','同','步']
tag_list = ['B-GPE','E-GPE','B-LOC','E-LOC','O','O','O','O','O','O','O','O','O']
data_sample = NERCnSample({'x': sent1, 'y': tag_list})
swap_ins = EntTypos()


class TestSpecialEntityTyposSwap(unittest.TestCase):
    def test_mode(self):
        self.assertTrue(swap_ins.mode in ['random', 'replace','swap'])

    def test_get_typo_method(self):
        self.assertTrue(swap_ins._get_typo_method() in [_replace_cn, swap])
        swap_ins.mode = 'replace'
        self.assertTrue(swap_ins._get_typo_method() == _replace_cn)
        swap_ins.mode = 'swap'
        self.assertTrue(swap_ins._get_typo_method() == swap)


    def test_get_replacement_words(self):
        pass

    def test_EntityTyposSwap(self):
        # TODO 基类会增加transform 输入校验
        # self.assertRaises(AssertionError, swap_ins.transform, '')
        # self.assertRaises(AssertionError, swap_ins.transform, [])
        # self.assertRaises(AssertionError, swap_ins.transform, SASample)

        # test data with no entity
        data = NERCnSample({'x': data_sample.text.tokens, 'y': ['O'] *
                                        len(data_sample.text.tokens)})
        self.assertEqual([], swap_ins.transform(data))

        # test faction
        # change = swap_ins.transform(data_sample, n=5)
        # self.assertTrue(5 == len(change))
        # for item in change:
        #     for ori_word, trans_word, tags in zip(data_sample.get_words('text'),
        #                                           item.get_words('text'),
        #                                           data_sample.tags):
        #         self.assertTrue(ori_word == trans_word if tags == 'O'
        #                         else ori_word != trans_word)


if __name__ == "__main__":
    unittest.main()

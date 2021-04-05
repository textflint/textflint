import unittest
import editdistance

from TextFlint.input_layer.component.sample.ner_sample import NERSample
from TextFlint.generation_layer.transformation.NER.ent_typos import EntTypos
from TextFlint.common.utils.word_op import *

sent1 = 'EU rejects German call to boycott British lamb .'
data_sample = NERSample(
        {'x': sent1, 'y': ['B-ORG', 'O', 'B-MISC', 'O', 'O',
                           'O', 'B-MISC', 'O', 'O']})
swap_ins = EntTypos()


class TestSpecialEntityTyposSwap(unittest.TestCase):
    def test_mode(self):
        self.assertTrue(swap_ins.mode in ['random', 'replace',
                                          'swap', 'insert', 'delete'])

    def test_get_typo_method(self):
        self.assertTrue(swap_ins._get_typo_method() in [replace, swap,
                                                        insert, delete])
        swap_ins.mode = 'replace'
        self.assertTrue(swap_ins._get_typo_method() == replace)
        swap_ins.mode = 'swap'
        self.assertTrue(swap_ins._get_typo_method() == swap)
        swap_ins.mode = 'insert'
        self.assertTrue(swap_ins._get_typo_method() == insert)
        swap_ins.mode = 'delete'
        self.assertTrue(swap_ins._get_typo_method() == delete)

    def test_get_replacement_words(self):
        self.assertTrue([] == swap_ins._get_replacement_words(''))

        for i in range(10):
            self.assertTrue(editdistance.eval('qwertyuiop',
                        swap_ins._get_replacement_words('qwertyuiop')[0]) <= 2)

    def test_EntityTyposSwap(self):
        # TODO 基类会增加transform 输入校验
        # self.assertRaises(AssertionError, swap_ins.transform, '')
        # self.assertRaises(AssertionError, swap_ins.transform, [])
        # self.assertRaises(AssertionError, swap_ins.transform, SASample)

        # test data with no entity
        data = NERSample({'x': data_sample.get_words('text'), 'y': ['O'] *
                                        len(data_sample.get_words('text'))})
        self.assertEqual([], swap_ins.transform(data))

        # test faction
        change = swap_ins.transform(data_sample, n=5)
        self.assertTrue(5 == len(change))
        for item in change:
            for ori_word, trans_word, tags in zip(data_sample.get_words('text'),
                                                  item.get_words('text'),
                                                  data_sample.get_value('tags')):
                self.assertTrue(ori_word == trans_word if tags == 'O'
                                else ori_word != trans_word)


if __name__ == "__main__":
    unittest.main()

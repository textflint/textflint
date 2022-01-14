# -- coding: utf-8 --
import unittest
from textflint.input.component.sample.nercn_sample import NERCnSample
from textflint.generation.transformation.NERCN.swap_ent import SwapEnt

data = {'x':'上海浦东开发与法制建设同步','y':['B-GPE','E-GPE','B-LOC','E-LOC','O','O','O','O','O','O','O','O','O']}
data_sample = NERCnSample(data)


class TestSpecialSwapEnt(unittest.TestCase):
    def test_mode(self):
        swap_ins_oov = SwapEnt()
        self.assertTrue(swap_ins_oov.swap_type in ['OOV','SwapLonger'])

    def test_given_mode(self):
        # test with wrong mode
        self.assertRaises(ValueError, SwapEnt, 'qabc')

        # test with OOV mode
        swap_ins_oov = SwapEnt('OOV')
        self.assertTrue(swap_ins_oov.swap_type == 'OOV')

        # # test with CrossCategory mode
        # swap_ins_oov = SwapEnt('CrossCategory')
        # self.assertTrue(swap_ins_oov.swap_type == 'CrossCategory')

        # test with SwapLonger mode
        swap_ins_oov = SwapEnt('SwapLonger')
        self.assertTrue(swap_ins_oov.swap_type == 'SwapLonger')

    def test_SwapEnt(self):
        # test data with no entity
        swap_ins_oov = SwapEnt('OOV')
        data = NERCnSample({'x': data_sample.get_tokens('text'),
                          'y': ['O'] * len(data_sample.get_tokens('text'))})
        change = swap_ins_oov.transform(data, n=5)
        self.assertEqual([], change)


        # test tolonger
        # not necessarily exist

        # test multientity

        # test oov
        change = swap_ins_oov.transform(data_sample, n=5)
        self.assertTrue(5 >= len(change))
        for item in change:
            trans_entities = item.find_entities_BMOES(item.text.tokens, item.tags)
            ori_entities = item.find_entities_BMOES(
                data_sample.text.tokens, data_sample.tags)
            for ori_ent, trans_ent in zip(ori_entities, trans_entities):
                if ori_ent['tag'] in ['PER', 'LOC', 'ORG','GPE','NS','NR','NT']:
                    self.assertTrue(trans_ent['entity'] != ori_ent['entity'])


if __name__ == "__main__":
    unittest.main()

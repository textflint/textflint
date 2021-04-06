import unittest
from textflint.input_layer.component.sample.ner_sample import NERSample
from textflint.generation_layer.transformation.NER.swap_ent import SwapEnt

sent1 = 'SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .'
tags = []
data_sample = NERSample(
        {'x': sent1, 'y': ['O', 'O', 'B-LOC', 'O', 'O', 'O', 'O',
                           'B-PER', 'O', 'O', 'O', 'O']})


class TestSpecialSwapEnt(unittest.TestCase):
    def test_mode(self):
        swap_ins_oov = SwapEnt()
        self.assertTrue(swap_ins_oov.swap_type in ['CrossCategory', 'OOV',
                                                   'SwapLonger'])

    def test_given_mode(self):
        # test with wrong mode
        self.assertRaises(ValueError, SwapEnt, 'qabc')

        # test with OOV mode
        swap_ins_oov = SwapEnt('OOV')
        self.assertTrue(swap_ins_oov.swap_type == 'OOV')

        # test with CrossCategory mode
        swap_ins_oov = SwapEnt('CrossCategory')
        self.assertTrue(swap_ins_oov.swap_type == 'CrossCategory')

        # test with SwapLonger mode
        swap_ins_oov = SwapEnt('SwapLonger')
        self.assertTrue(swap_ins_oov.swap_type == 'SwapLonger')

    def test_SwapEnt(self):
        # test data with no entity
        swap_ins_oov = SwapEnt('OOV')
        data = NERSample({'x': data_sample.get_words('text'),
                          'y': ['O'] * len(data_sample.get_words('text'))})
        change = swap_ins_oov.transform(data, n=5)
        self.assertEqual([], change)

        # test tolonger
        swap_ins_longer = SwapEnt('SwapLonger')
        change = swap_ins_longer.transform(data_sample, n=5)
        self.assertTrue(5 >= len(change))
        for item in change:
            trans_entities = item.find_entities_BIO(item.text.words, item.tags)
            ori_entities = item.find_entities_BIO(
                data_sample.text.words, data_sample.tags)
            for ori_ent, trans_ent in zip(ori_entities, trans_entities):
                if ori_ent['tag'] in ['PER', 'LOC', 'ORG']:
                    self.assertTrue(len(trans_ent['entity'].split(' ')) >= 3)

        # test multientity
        swap_ins_cross = SwapEnt('CrossCategory')
        change = swap_ins_cross.transform(data_sample, n=5)
        self.assertTrue(5 >= len(change))
        for item in change:
            trans_entities = item.find_entities_BIO(item.text.words, item.tags)
            ori_entities = item.find_entities_BIO(
                data_sample.text.words, data_sample.tags)
            for ori_ent, trans_ent in zip(ori_entities, trans_entities):
                if ori_ent['tag'] in ['PER', 'LOC', 'ORG']:
                    self.assertTrue(trans_ent['entity'] != ori_ent['entity'])

        # test oov
        change = swap_ins_oov.transform(data_sample, n=5)
        self.assertTrue(5 >= len(change))
        for item in change:
            trans_entities = item.find_entities_BIO(item.text.words, item.tags)
            ori_entities = item.find_entities_BIO(
                data_sample.text.words, data_sample.tags)
            for ori_ent, trans_ent in zip(ori_entities, trans_entities):
                if ori_ent['tag'] in ['PER', 'LOC', 'ORG']:
                    self.assertTrue(trans_ent['entity'] != ori_ent['entity'])


if __name__ == "__main__":
    unittest.main()

import unittest

from textflint.input_layer.component.sample import SASample
from textflint.generation_layer.transformation.UT.swap_named_ent \
    import SwapNamedEnt


class TestEntity(unittest.TestCase):
    def test_transformation(self):
        sent1 = 'Lionel Messi is a football player from Argentina. ' \
                'Fudan University is located in Shanghai province, ' \
                'Alibaba with 50000 staff. Wang Xiao is a stuendent. ' \
                'Zhangheng road in Pudong area.'
        data_sample = SASample({'x': sent1, 'y': "positive"})
        swap_ins = SwapNamedEnt()

        # test decompose_entities_info
        words = data_sample.get_words('x')
        a, b, c = swap_ins.decompose_entities_info(data_sample.get_ner('x'))
        for pos, word, label in zip(a, b, c):
            self.assertTrue(label in ['LOCATION', 'PERSON', 'ORGANIZATION'])
            self.assertEqual(words[pos[0]:pos[1]], word.split(' '))

        # test transformation
        import random
        random.seed(208)

        trans = swap_ins.transform(data_sample, n=5)
        self.assertEqual(5, len(trans))
        change = ["Mr Ross is a football player from Tashkent. Fudan Unive"
                  "rsity is located in South Zone province, "
                  "Zagreb with 50000 staff. Jean Chrétien is a stuendent. Zhan"
                  "gheng road in  Czech Republic area.",
                  "Mr Ross is a football player from Tashkent. Fudan Univer"
                  "sity is located in South Zone province, "
                  "Zagreb with 50000 staff. Jean Chrétien is a stuendent. "
                  "Zhangheng road in the valley area.",
                  "Mr Ross is a football player from Tashkent. Fudan University"
                  " is located in South Zone province, "
                  "Zagreb with 50000 staff. Jean Chrétien is a stuendent. "
                  "Zhangheng road in Parvan area.",
                  "Mr Ross is a football player from Tashkent. Fudan University"
                  " is located in South Zone province, "
                  "Zagreb with 50000 staff. "
                  "Jean Chrétien is a stuendent. Zhangheng road in East-West "
                  "area.",
                  "Mr Ross is a football player from Tashkent. Fudan "
                  "University is located in east Atlantic province, "
                  "Prague with 50000 staff. Mr Mayoral is a stuendent. "
                  "Zhangheng road in West Midlands area."]
        for sample, sent in zip(trans, change):
            self.assertTrue("positive", sample.get_value('y'))
            self.assertEqual(sent, sample.get_text('x'))

        # test special sample
        special_sample = SASample({'x': '', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))
        special_sample = SASample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], swap_ins.transform(special_sample))


if __name__ == "__main__":
    unittest.main()

import unittest
from textflint.input.component.sample.ner_sample import NERSample
from textflint.generation.transformation.NER.concat_sent import ConcatSent

sent1 = 'EU rejects German call to boycott British lamb .'
sent_list = ['EU', 'rejects', 'German', 'call', 'to', 'boycott',
             'British', 'lamb', '.']
tag_list = ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
data_sample = NERSample({'x': sent1, 'y': tag_list})
concat_samples = [NERSample({'x': sent1, 'y': tag_list})]
swap_ins = ConcatSent()


class TestSpecialEntityTyposSwap(unittest.TestCase):

    def test_ConcatCase(self):
        # TODO 基类会增加transform 输入校验
        # self.assertRaises(AssertionError, swap_ins.transform, '')
        # self.assertRaises(AssertionError, swap_ins.transform, [])
        # self.assertRaises(AssertionError, swap_ins.transform, SASample)

        # test empty concat samples
        change = swap_ins.transform(data_sample, n=1, concat_samples=[])
        self.assertEqual(change, [])

        # test faction
        change = swap_ins.transform(data_sample, n=1,
                                    concat_samples=concat_samples)
        self.assertEqual(change[0].get_words('text'),
                         data_sample.get_words('text') +
                         concat_samples[0].get_words('text'))
        self.assertEqual(change[0].get_value('tags'),
                         data_sample.get_value('tags') +
                         concat_samples[0].get_value('tags'))


if __name__ == "__main__":
    unittest.main()

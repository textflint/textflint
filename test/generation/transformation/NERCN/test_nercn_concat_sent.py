import unittest
from textflint.input.component.sample.nercn_sample import NERCnSample
from textflint.generation.transformation.NERCN.concat_sent import ConcatSent

sent1 = '上海浦东开发与经济建设同步'
sent_list = ['上','海','浦','东','开','发','与','法','制','建','设','同','步']
tag_list = ['B-GPE','E-GPE','B-LOC','E-LOC','O','O','O','O','O','O','O','O','O']
data_sample = NERCnSample({'x': sent1, 'y': tag_list})
concat_samples = [NERCnSample({'x': sent1, 'y': tag_list})]
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
        self.assertEqual(change[0].get_tokens('text'),
                         data_sample.get_tokens('text') +
                         concat_samples[0].get_tokens('text'))

        self.assertEqual(change[0].tags.field_value,
                         data_sample.tags.field_value +
                         concat_samples[0].tags.field_value)


if __name__ == "__main__":
    unittest.main()
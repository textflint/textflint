import unittest

from textflint.input.dataset import Dataset
from textflint.generation.generator.smcn_generator import SMCNGenerator

sample1 = {'sentence1': '天气预报说上海今天下雨的概率是50%。',
        'sentence2': '根据天气预报，上海今天有80%的概率下雨。',
        'y': '0'}
sample2 = {'sentence1': '我喜欢这本书。',
        'sentence2': '这本书是我最喜欢的。',
        'y': '1'}

sample3 = {'sentence1': '! @ # $ % ^ & * ( )',
        'sentence2': '! @ # $ % ^ & * ( )',
        'y': '0'}

data_samples = [sample1, sample2, sample3]
dataset = Dataset(task='SMCN')
dataset.load(data_samples)
gene = SMCNGenerator()


class TestSMCNGenerator(unittest.TestCase):

    def test_generate(self):
        # test task transformation, ignore SMCNOverlap because
        # it does't rely on the original data

        trans_methods = ["SwapWord", "SwapNum"]
        gene = SMCNGenerator(trans_methods=trans_methods,
                           sub_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            for index in range(len(original_samples)):
                # test whether the sample changed or not
                self.assertTrue(original_samples[index].sentence1.field_value
                                != trans_rst[index].sentence1.field_value or
                                original_samples[index].sentence2.field_value
                                != trans_rst[index].sentence2.field_value)

                # SmAntonymSwap makes the label to be '0'
                if trans_type == 'SwapWord':
                    self.assertEqual('0', trans_rst[index].y.field_value)
        # test wrong trans_methods
        gene = SMCNGenerator(trans_methods=["wrong_transform_method"],
                           sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = SMCNGenerator(trans_methods=["AddSubtree"],
                           sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = SMCNGenerator(trans_methods="OOV",
                           sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))


if __name__ == "__main__":
    unittest.main()

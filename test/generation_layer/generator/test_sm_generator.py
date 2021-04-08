import unittest

from textflint.input_layer.dataset import Dataset
from textflint.generation_layer.generator.sm_generator import SMGenerator

sample1 = {'sentence1': 'MR zhang has 10 students',
        'sentence2': 'Mr zhang has 20 students',
        'y': '0'}
sample2 = {'sentence1': 'I like eating apples',
        'sentence2': 'I love to eat apples',
        'y': '1'}
sample3 = {'sentence1': 'There are two little boys smiling.',
        'sentence2': 'Two little boys are smiling and laughing '
                     'while one is standing and one is in a bouncy seat',
        'y': '0'}

sample4 = {'sentence1': '! @ # $ % ^ & * ( )',
        'sentence2': '! @ # $ % ^ & * ( )',
        'y': '0'}

data_samples = [sample1, sample2, sample3, sample4]
dataset = Dataset(task='SM')
dataset.load(data_samples)
gene = SMGenerator()


class TestSMGenerator(unittest.TestCase):

    def test_generate(self):
        # test task transformation, ignore SmOverlap because
        # it does't rely on the original data
        transformation_methods = ["SwapWord", "SwapNum"]
        gene = SMGenerator(transformation_methods=transformation_methods,
                           subpopulation_methods=[])
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

        # test part of UT transformations
        gene = SMGenerator(transformation_methods=['WordCase'],
                           subpopulation_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(4, len(original_samples))
            for index in range(len(original_samples)):
                for trans_word, ori_word in zip(
                        trans_rst[index].get_words('sentence1'),
                        original_samples[index].get_words('sentence1')):
                    self.assertEqual(trans_word, ori_word.lower())
                for trans_word, ori_word in zip(
                        trans_rst[index].get_words('sentence2'),
                        original_samples[index].get_words('sentence2')):
                    self.assertEqual(trans_word, ori_word.upper())
        gene = SMGenerator(transformation_methods=['SwapNum'],
                           subpopulation_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(1, len(original_samples))
            for index in range(len(original_samples)):
                for trans_word, ori_word in \
                        zip(trans_rst[index].get_words('sentence1'),
                            original_samples[index].get_words('sentence1')):
                    if ori_word.isdigit():
                        self.assertTrue(ori_word != trans_word)
                for trans_word, ori_word in \
                        zip(trans_rst[index].get_words('sentence2'),
                            original_samples[index].get_words('sentence2')):
                    if ori_word.isdigit():
                        self.assertTrue(ori_word != trans_word)

        # test wrong transformation_methods
        gene = SMGenerator(transformation_methods=["wrong_transform_method"],
                           subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = SMGenerator(transformation_methods=["AddSubtree"],
                           subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = SMGenerator(transformation_methods="OOV",
                           subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))


if __name__ == "__main__":
    unittest.main()
